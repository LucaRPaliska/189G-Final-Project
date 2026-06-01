# ============================================================
# VLM Hallucination Probe — Feature Extraction
#
# This script:
#   1. Loads POPE yes/no benchmark samples from HuggingFace
#   2. Runs frozen VLM inference
#   3. Extracts hidden states from ALL layers
#   4. Labels samples as factual / nonfactual
#   5. Saves probe dataset
#
# Labeling modes (--labeling):
#   logit  [default] compare logit(yes) vs logit(no) at last prompt token
#   em     generate text, check against gold with Exact Match
#
# Hidden state modes (--hidden_state):
#   prompt [default] hidden state at last token of the prompt
#   output hidden state at last token of the generated answer (teacher-forcing)
#
# Commands:
# python3 extract_probe_features.py --model Qwen/Qwen2.5-VL-3B-Instruct --pooling last
# python3 extract_probe_features.py --model Qwen/Qwen2.5-VL-3B-Instruct --pooling last --labeling em --hidden_state output --few_shot_n 3
# python3 extract_probe_features.py --model OpenGVLab/InternVL2-2B --pooling last
# python3 extract_probe_features.py --model microsoft/Phi-3.5-vision-instruct --pooling last
# python3 extract_probe_features.py --model deepseek-ai/deepseek-vl-1.3b-base --pooling last
# python3 extract_probe_features.py --model openbmb/MiniCPM-V-2 --pooling last
# ============================================================

import os
import json
import argparse
import torch
import numpy as np

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str,   default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--pooling",      type=str,   default="last", choices=["last", "mean"])
parser.add_argument("--labeling",     type=str,   default="logit", choices=["logit", "em"],
                    help="logit: compare yes/no logits  em: generate + exact match")
parser.add_argument("--hidden_state", type=str,   default="prompt", choices=["prompt", "output"],
                    help="prompt: last prompt token  output: last output token (teacher-forcing)")
parser.add_argument("--few_shot_n",   type=int,   default=0,
                    help="Number of text few-shot examples to prepend (0 = direct question)")
parser.add_argument("--split",        type=str,   default="all",
                    choices=["adversarial", "popular", "random", "all"])
parser.add_argument("--shared_test",  action="store_true",
                    help="Save/load a shared test split (test_ids.json) for cross-model comparison")
parser.add_argument("--balance",      action="store_true",
                    help="Balance output to 50/50 factual/nonfactual")
parser.add_argument("--output",       type=str,   default=None)
parser.add_argument("--max_samples",  type=int,   default=None)
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME   = args.model
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.float16
MAX_SAMPLES  = args.max_samples
POOLING      = args.pooling
LABELING     = args.labeling
HIDDEN_STATE = args.hidden_state
FEW_SHOT_N   = args.few_shot_n

model_short = MODEL_NAME.replace("/", "_")

if args.output is not None:
    OUTPUT_FILE = args.output
else:
    OUTPUT_FILE = f"probe_dataset_{model_short}_{POOLING}_{LABELING}_{HIDDEN_STATE}.npz"

TEST_IDS_FILE = "test_ids.json"

# Few-shot examples (used when FEW_SHOT_N > 0)
_FEW_SHOT_POOL = [
    ("Is there a dog in the image?",           "No"),
    ("Is there a chair in the image?",         "Yes"),
    ("Is there an airplane in the image?",     "No"),
    ("Is there a cup in the image?",           "Yes"),
    ("Is there a fire hydrant in the image?",  "No"),
]
FEW_SHOT_EXAMPLES = _FEW_SHOT_POOL[:FEW_SHOT_N]

# ============================================================
# MODEL FAMILY DETECTION
# ============================================================

def detect_family(name):
    n = name.lower()
    if "qwen2.5-vl" in n or "qwen2-vl" in n: return "qwen2vl"
    if "phi-3.5-vision" in n or "phi-3-vision" in n: return "phi3vision"
    if "internvl" in n: return "internvl"
    if "deepseek-vl" in n: return "deepseek"
    if "minicpm" in n: return "minicpm"
    return "generic"

FAMILY = detect_family(MODEL_NAME)
print("Model family:", FAMILY)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

from transformers import AutoProcessor, AutoTokenizer

if FAMILY == "qwen2vl":
    from transformers import Qwen2_5_VLForConditionalGeneration
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

elif FAMILY == "phi3vision":
    from transformers import AutoModelForCausalLM
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True, _attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(
                    MODEL_NAME, trust_remote_code=True, num_crops=4)
    tokenizer = processor.tokenizer

elif FAMILY == "internvl":
    from transformers import AutoModel
    model     = AutoModel.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, trust_remote_code=True, use_fast=False)
    processor = None

elif FAMILY == "deepseek":
    from transformers import AutoModelForCausalLM
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer

elif FAMILY == "minicpm":
    from transformers import AutoModel
    model     = AutoModel.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = None

else:
    from transformers import AutoModelForCausalLM
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer

model.eval()

print("Model loaded.")

# ============================================================
# TOKEN IDS  (used for logit labeling)
# ============================================================

YES_ID = tokenizer.encode("yes", add_special_tokens=False)[0]
NO_ID  = tokenizer.encode("no",  add_special_tokens=False)[0]

print("YES token id:", YES_ID)
print("NO token id:", NO_ID)

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading POPE dataset...")

ds     = load_dataset("lmms-lab/POPE", split="test")
import pandas as pd
df_all = ds.to_pandas()

if args.split != "all":
    df_all = df_all[df_all["category"] == args.split].reset_index(drop=True)

if MAX_SAMPLES is not None:
    df_all = df_all.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

print(f"Loaded {len(df_all)} samples.")

# ============================================================
# SHARED TEST SPLIT
# ============================================================

if args.shared_test:

    if os.path.exists(TEST_IDS_FILE):
        with open(TEST_IDS_FILE) as f:
            test_ids = set(json.load(f))
        print(f"Loaded {len(test_ids)} shared test IDs from {TEST_IDS_FILE}")

    else:
        rng  = np.random.default_rng(42)
        n_per = int(len(df_all) * 0.2) // 2
        yes_q = df_all[df_all["answer"] == "yes"]["question_id"].astype(str).tolist()
        no_q  = df_all[df_all["answer"] == "no" ]["question_id"].astype(str).tolist()
        test_ids = set(
            rng.choice(yes_q, size=min(n_per, len(yes_q)), replace=False).tolist()
            + rng.choice(no_q, size=min(n_per, len(no_q)),  replace=False).tolist()
        )
        with open(TEST_IDS_FILE, "w") as f:
            json.dump(list(test_ids), f)
        print(f"Created shared test split: {len(test_ids)} IDs saved to {TEST_IDS_FILE}")

else:
    test_ids = set()

# ============================================================
# HELPERS
# ============================================================

def get_pil_image(img_field):
    if hasattr(img_field, "convert"):
        return img_field.convert("RGB")
    if isinstance(img_field, dict):
        if img_field.get("bytes") is not None:
            return Image.open(BytesIO(img_field["bytes"])).convert("RGB")
        if img_field.get("path"):
            return Image.open(img_field["path"]).convert("RGB")
    if isinstance(img_field, bytes):
        return Image.open(BytesIO(img_field)).convert("RGB")
    return None


def build_prompt(question):
    # With few-shot examples
    if FEW_SHOT_N > 0:
        lines = ["Here are some example questions and answers about images:"]
        for q, a in FEW_SHOT_EXAMPLES:
            lines.append(f"Q: {q}  A: {a}.")
        lines += ["", "Now answer the following question. Reply with only 'yes' or 'no'.", f"Q: {question}"]
        return "\n".join(lines)
    # Direct question (Gege default)
    return question + " Answer only yes or no."


def exact_match(pred, gold):
    p = pred.strip().lower().rstrip(".,!?")
    g = gold.strip().lower().rstrip(".,!?")
    return p == g or p.startswith(g)


def encode_inputs(image, question, answer=None):
    """
    Encode image + question into model input tensors.
    If answer is provided, appends it as the assistant turn (for teacher-forcing).
    Returns dict of tensors on model.device.
    """

    prompt = build_prompt(question)

    # --------------------------------------------------------
    if FAMILY == "qwen2vl":

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt},
        ]}]

        if answer is not None:
            messages.append({"role": "assistant", "content": answer})

        text   = processor.apply_chat_template(
                     messages, tokenize=False,
                     add_generation_prompt=(answer is None))
        inputs = processor(text=[text], images=[image],
                           padding=True, return_tensors="pt")

    # --------------------------------------------------------
    elif FAMILY == "phi3vision":

        answer_part = answer if answer is not None else ""
        end_tag     = "<|end|>" if answer is not None else ""
        text        = (f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n"
                       f"<|assistant|>{answer_part}{end_tag}")
        inputs      = processor(text=text, images=[image], return_tensors="pt")

    # --------------------------------------------------------
    elif FAMILY == "internvl":

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        pixel_values = transform(image).unsqueeze(0).to(DTYPE).to(model.device)
        text         = f"<image>\n{prompt}"

        if answer is not None:
            text += f"\n<|im_start|>assistant\n{answer}<|im_end|>"

        ids    = tokenizer(text, return_tensors="pt")
        inputs = {
            "pixel_values":  pixel_values,
            "input_ids":     ids["input_ids"].to(model.device),
            "attention_mask":ids["attention_mask"].to(model.device),
        }
        return inputs

    # --------------------------------------------------------
    elif FAMILY == "deepseek":

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]

        if answer is not None:
            messages.append({"role": "assistant", "content": answer})

        text   = processor.apply_chat_template(
                     messages, tokenize=False,
                     add_generation_prompt=(answer is None))
        inputs = processor(text=[text], images=[image], return_tensors="pt")

    # --------------------------------------------------------
    else:

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt},
        ]}]

        if answer is not None:
            messages.append({"role": "assistant", "content": answer})

        text   = processor.apply_chat_template(
                     messages, tokenize=False,
                     add_generation_prompt=(answer is None))
        inputs = processor(text=[text], images=[image],
                           padding=True, return_tensors="pt")

    # --------------------------------------------------------

    return {k: v.to(model.device) for k, v in inputs.items()}

# ============================================================
# STORAGE
# ============================================================

all_features = []
all_labels   = []
all_preds    = []
all_gts      = []
all_ids      = []
all_splits   = []

# ============================================================
# MAIN LOOP
# ============================================================

for _, row in tqdm(df_all.iterrows(), total=len(df_all)):

    # --------------------------------------------------------
    # LOAD SAMPLE
    # --------------------------------------------------------

    question_id = str(row["question_id"])
    question    = row["question"]
    gt          = row["answer"].strip().lower()
    split_tag   = "test" if question_id in test_ids else "train_val"

    image = get_pil_image(row["image"])

    if image is None:
        print(f"Warning: no image for question_id={question_id}, skipping.")
        continue

    try:

        # --------------------------------------------------------
        # ENCODE PROMPT + FORWARD PASS
        # --------------------------------------------------------

        prompt_inputs = encode_inputs(image, question)
        prompt_len    = prompt_inputs["input_ids"].shape[1]

        fwd_kwargs = {k: v for k, v in prompt_inputs.items()
                      if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            prompt_outputs = model(
                **fwd_kwargs,
                output_hidden_states=True,
                return_dict=True,
            )

        # --------------------------------------------------------
        # LABELING
        # --------------------------------------------------------

        if LABELING == "logit":

            # logits shape: [batch, seq_len, vocab]
            final_logits = prompt_outputs.logits[0, -1]

            yes_logit = final_logits[YES_ID]
            no_logit  = final_logits[NO_ID]

            pred  = "yes" if yes_logit > no_logit else "no"
            label = int(pred != gt)

        else:  # em

            gen_kwargs = {k: v for k, v in prompt_inputs.items()
                          if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                gen_ids = model.generate(
                    **gen_kwargs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            pred  = tokenizer.decode(
                        gen_ids[0, prompt_len:],
                        skip_special_tokens=True,
                    ).strip().lower()
            label = int(not exact_match(pred, gt))

        # --------------------------------------------------------
        # HIDDEN STATE SOURCE
        # --------------------------------------------------------

        if HIDDEN_STATE == "prompt":

            # Use hidden states from the prompt forward pass above
            feature_outputs = prompt_outputs

            # n_new not meaningful here; pooling is over prompt tokens
            n_new = 1

        else:  # output — teacher-forcing pass on (prompt + answer)

            full_inputs = encode_inputs(image, question, answer=pred)
            full_len    = full_inputs["input_ids"].shape[1]
            n_new       = max(full_len - prompt_len, 1)

            full_kwargs = {k: v for k, v in full_inputs.items()
                           if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                feature_outputs = model(
                    **full_kwargs,
                    output_hidden_states=True,
                    return_dict=True,
                )

        # --------------------------------------------------------
        # EXTRACT FEATURES
        # --------------------------------------------------------

        # hidden_states: tuple of (num_layers + 1) tensors
        # each: [batch, seq_len, hidden_dim]

        hidden_states = feature_outputs.hidden_states

        layer_features = []

        for layer_idx in range(len(hidden_states)):

            h = hidden_states[layer_idx][0]  # [seq_len, hidden_dim]

            if POOLING == "last":
                vec = h[-1, :].float().cpu().numpy()
            else:
                if HIDDEN_STATE == "output":
                    vec = h[-n_new:, :].float().mean(dim=0).cpu().numpy()
                else:
                    vec = h.float().mean(dim=0).cpu().numpy()

            layer_features.append(vec)

        layer_features = np.stack(layer_features)

        # shape: [num_layers+1, hidden_dim]

    except Exception as e:
        print(f"Warning: skipping question_id={question_id}: {e}")
        continue

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_features.append(layer_features)
    all_labels.append(label)
    all_preds.append(pred)
    all_gts.append(gt)
    all_ids.append(question_id)
    all_splits.append(split_tag)

# ============================================================
# BALANCE  (optional)
# ============================================================

all_features = np.stack(all_features)
all_labels   = np.array(all_labels)
all_splits   = np.array(all_splits)

if args.balance:

    tv_mask  = all_splits != "test"
    tv_idx   = np.where(tv_mask)[0]
    te_idx   = np.where(~tv_mask)[0]

    pos_idx  = tv_idx[all_labels[tv_idx] == 1]
    neg_idx  = tv_idx[all_labels[tv_idx] == 0]
    n_min    = min(len(pos_idx), len(neg_idx))

    rng      = np.random.default_rng(42)
    sel      = np.concatenate([
                   rng.choice(pos_idx, n_min, replace=False),
                   rng.choice(neg_idx, n_min, replace=False),
               ])
    rng.shuffle(sel)

    keep         = np.sort(np.concatenate([sel, te_idx]))
    all_features = all_features[keep]
    all_labels   = all_labels[keep]
    all_preds    = [all_preds[i]  for i in keep]
    all_gts      = [all_gts[i]    for i in keep]
    all_ids      = [all_ids[i]    for i in keep]
    all_splits   = all_splits[keep]

    print(f"Balanced to {n_min} per class ({2*n_min} train_val samples).")

# ============================================================
# SAVE
# ============================================================

print(f"Features ({POOLING}) shape:", all_features.shape)
print("Labels shape:", all_labels.shape)

unique, counts = np.unique(all_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label={u}: {c}  ({100*c/len(all_labels):.1f}%)")

save_kwargs = dict(
    features = all_features,
    labels   = all_labels,
    preds    = np.array(all_preds),
    gts      = np.array(all_gts),
)

if args.shared_test:
    save_kwargs["ids"]    = np.array(all_ids)
    save_kwargs["splits"] = all_splits

np.savez(OUTPUT_FILE, **save_kwargs)

print(f"Saved to {OUTPUT_FILE}")

# ============================================================
# SHAPES
# ============================================================

# features: [N, num_layers+1, hidden_dim]
# labels:   [N]  0=factual, 1=nonfactual
#
# Example:
# N=5000, layers=29, hidden_dim=2048 => [5000, 29, 2048]
#
# In train script:
# X = features[:, layer_idx, :]
#
# ============================================================
