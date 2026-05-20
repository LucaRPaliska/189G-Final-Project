"""
collect_hidden_states.py

Collects VLM hidden states on POPE questions for probe training.
Adapted from the LM probe paper method for VLMs.

Key differences from logit-based extraction (Gege's extract_probe_features.py):
  - Uses generation + Exact Match (EM) for factuality labeling
  - Collects hidden states at the OUTPUT token position via teacher-forcing
    (re-encodes input + model's generated answer, takes last-position hidden state)
  - Uses text few-shot demonstrations in the prompt
  - Creates a shared test split file (test_ids.json) for cross-model comparison

Supported models:
  Qwen/Qwen2.5-VL-3B-Instruct       (primary, full support)
  microsoft/Phi-3.5-vision-instruct  (full support)
  OpenGVLab/InternVL2-2B             (full support)
  deepseek-ai/deepseek-vl-1.3b-base  (full support, matches Gege)
  deepseek-ai/deepseek-vl2-tiny      (also supported)
  openbmb/MiniCPM-V-2                (full support)

Data source: Luca/POPE/Full/ parquet files (adversarial / popular / random)

Usage:
  python collect_hidden_states.py --model Qwen/Qwen2.5-VL-3B-Instruct
  python collect_hidden_states.py --model microsoft/Phi-3.5-vision-instruct
  python collect_hidden_states.py --model OpenGVLab/InternVL2-2B
  python collect_hidden_states.py --model deepseek-ai/deepseek-vl2-tiny
  python collect_hidden_states.py --model openbmb/MiniCPM-V-2
  python collect_hidden_states.py --model Qwen/Qwen2.5-VL-3B-Instruct --split popular --max_samples 500
"""

import os
import json
import argparse
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument(
    "--split", type=str, default="all",
    choices=["adversarial", "popular", "random", "all"],
)
parser.add_argument(
    "--pooling", type=str, default="last", choices=["last", "mean"],
    help="'last' = last output token, 'mean' = mean over all new tokens",
)
parser.add_argument("--max_samples",   type=int,   default=None)
parser.add_argument("--few_shot_n",    type=int,   default=3)
parser.add_argument("--test_ids_file", type=str,   default=None)
parser.add_argument("--test_fraction", type=float, default=0.2)
parser.add_argument("--output",        type=str,   default=None)
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = args.model
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16   # bfloat16 is safer across model families than float16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POPE_DIR   = os.path.join(SCRIPT_DIR, "POPE", "Full")

model_short  = MODEL_NAME.replace("/", "_")
OUTPUT_FILE  = args.output or os.path.join(
    SCRIPT_DIR,
    f"hidden_states_{model_short}_{args.split}_{args.pooling}.npz",
)
TEST_IDS_FILE = args.test_ids_file or os.path.join(SCRIPT_DIR, "test_ids.json")

_ALL_FEW_SHOT = [
    ("Is there a dog in the image?",           "No"),
    ("Is there a chair in the image?",         "Yes"),
    ("Is there an airplane in the image?",     "No"),
    ("Is there a cup in the image?",           "Yes"),
    ("Is there a fire hydrant in the image?",  "No"),
]
FEW_SHOT_EXAMPLES = _ALL_FEW_SHOT[: args.few_shot_n]

# ============================================================
# MODEL FAMILY DETECTION
# ============================================================

def detect_family(name: str) -> str:
    n = name.lower()
    if "qwen2.5-vl" in n or "qwen2-vl" in n:
        return "qwen2vl"
    if "phi-3.5-vision" in n or "phi-3-vision" in n:
        return "phi3vision"
    if "internvl" in n:
        return "internvl"
    if "deepseek-vl" in n:
        return "deepseek"
    if "minicpm" in n:
        return "minicpm"
    return "generic"

FAMILY = detect_family(MODEL_NAME)
print(f"Detected model family: {FAMILY}")

# ============================================================
# MODEL LOADING  (family-specific)
# ============================================================

print(f"Loading model: {MODEL_NAME}")

if FAMILY == "qwen2vl":
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

elif FAMILY == "phi3vision":
    from transformers import AutoModelForCausalLM, AutoProcessor
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True, _attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(
                    MODEL_NAME, trust_remote_code=True, num_crops=4)
    tokenizer = processor.tokenizer

elif FAMILY == "internvl":
    from transformers import AutoModel, AutoTokenizer
    model     = AutoModel.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, trust_remote_code=True, use_fast=False)
    processor = None   # InternVL uses tokenizer directly

elif FAMILY == "deepseek":
    from transformers import AutoModelForCausalLM, AutoProcessor
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer

elif FAMILY == "minicpm":
    from transformers import AutoModel, AutoTokenizer
    model     = AutoModel.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = None

else:
    from transformers import AutoModelForCausalLM, AutoProcessor
    model     = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
                    trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer

model.eval()
print("Model loaded.")

# ============================================================
# LOAD DATASET
# ============================================================

print(f"Loading POPE Full data (split={args.split}) ...")
splits_to_load = (
    ["adversarial", "popular", "random"] if args.split == "all" else [args.split]
)
dfs = []
for s in splits_to_load:
    df = pd.read_parquet(os.path.join(POPE_DIR, f"{s}-00000-of-00001.parquet"))
    df["pope_split"] = s
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df_all)} samples.")
print(f"Answer distribution:\n{df_all['answer'].value_counts()}")

if args.max_samples is not None:
    df_all = df_all.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {len(df_all)} samples.")

# ============================================================
# SHARED TEST SPLIT
# ============================================================

if os.path.exists(TEST_IDS_FILE):
    with open(TEST_IDS_FILE) as f:
        test_ids = set(json.load(f))
    print(f"Loaded {len(test_ids)} shared test IDs from {TEST_IDS_FILE}")
else:
    n_test   = int(len(df_all) * args.test_fraction)
    yes_qids = df_all[df_all["answer"] == "yes"]["question_id"].astype(str).tolist()
    no_qids  = df_all[df_all["answer"] == "no" ]["question_id"].astype(str).tolist()
    rng      = np.random.default_rng(42)
    n_per    = n_test // 2
    test_ids = set(
        rng.choice(yes_qids, size=min(n_per, len(yes_qids)), replace=False).tolist()
        + rng.choice(no_qids,  size=min(n_per, len(no_qids)),  replace=False).tolist()
    )
    with open(TEST_IDS_FILE, "w") as f:
        json.dump(list(test_ids), f)
    print(f"Created shared test split: {len(test_ids)} IDs → {TEST_IDS_FILE}")

# ============================================================
# HELPERS
# ============================================================

def get_pil_image(img_field) -> Image.Image:
    if img_field is None:
        return None
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


def exact_match(pred: str, gold: str) -> bool:
    p = pred.strip().lower().rstrip(".,!?")
    g = gold.strip().lower().rstrip(".,!?")
    return p == g or p.startswith(g + " ") or p.startswith(g)


def build_user_prompt(question: str) -> str:
    lines = ["Here are some example questions and answers about images:"]
    for q, a in FEW_SHOT_EXAMPLES:
        lines.append(f"Q: {q}  A: {a}.")
    lines += [
        "",
        "Now answer the following question about the provided image.",
        "Reply with only 'yes' or 'no'.",
        f"Q: {question}",
    ]
    return "\n".join(lines)

# ============================================================
# FAMILY-SPECIFIC: ENCODE INPUTS
# ============================================================

def encode_prompt(image: Image.Image, question: str) -> dict:
    """Encode (image, prompt) → model input tensors. Family-specific."""
    prompt_text = build_user_prompt(question)

    if FAMILY == "qwen2vl":
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ]}]
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image],
                           padding=True, return_tensors="pt")

    elif FAMILY == "phi3vision":
        # Phi-3.5-vision uses <|image_1|> placeholder in text
        phi_prompt = (
            f"<|user|>\n<|image_1|>\n{prompt_text}<|end|>\n<|assistant|>"
        )
        inputs = processor(text=phi_prompt, images=[image], return_tensors="pt")

    elif FAMILY == "internvl":
        # InternVL2 uses pixel_values + text ids separately
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = transform(image).unsqueeze(0).to(DTYPE).to(model.device)
        # InternVL wraps question with <image> token
        intern_question = f"<image>\n{prompt_text}"
        text_ids = tokenizer(intern_question, return_tensors="pt")
        inputs = {
            "pixel_values":  pixel_values,
            "input_ids":     text_ids["input_ids"].to(model.device),
            "attention_mask":text_ids["attention_mask"].to(model.device),
            "_internvl_question": intern_question,   # kept for full-seq re-encode
        }

    elif FAMILY == "deepseek":
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], return_tensors="pt")

    elif FAMILY == "minicpm":
        msgs = [{"role": "user", "content": prompt_text}]
        # MiniCPM's tokenizer handles image via special tokens
        full_text, inputs_dict = model.build_conversation_input_ids(
            tokenizer, query=prompt_text, image=image)
        inputs = {k: v.unsqueeze(0).to(model.device) if v.dim() == 0
                  else v.unsqueeze(0).to(model.device)
                  for k, v in inputs_dict.items()}

    else:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ]}]
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image],
                           padding=True, return_tensors="pt")

    # Move to device (skip private helper keys starting with _)
    return {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def encode_full(image: Image.Image, question: str, answer: str) -> dict:
    """Re-encode (image, prompt + answer) for teacher-forcing. Family-specific."""
    prompt_text = build_user_prompt(question)

    if FAMILY == "qwen2vl":
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ]},
            {"role": "assistant", "content": answer},
        ]
        chat   = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[chat], images=[image],
                           padding=True, return_tensors="pt")

    elif FAMILY == "phi3vision":
        phi_prompt = (
            f"<|user|>\n<|image_1|>\n{prompt_text}<|end|>\n"
            f"<|assistant|>{answer}<|end|>"
        )
        inputs = processor(text=phi_prompt, images=[image], return_tensors="pt")

    elif FAMILY == "internvl":
        # Append answer after the question with role markers
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values  = transform(image).unsqueeze(0).to(DTYPE).to(model.device)
        full_text     = (
            f"<image>\n{prompt_text}"
            f"\n<|im_start|>assistant\n{answer}<|im_end|>"
        )
        text_ids = tokenizer(full_text, return_tensors="pt")
        inputs   = {
            "pixel_values":  pixel_values,
            "input_ids":     text_ids["input_ids"].to(model.device),
            "attention_mask":text_ids["attention_mask"].to(model.device),
        }

    elif FAMILY == "deepseek":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": answer},
        ]
        chat   = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[chat], images=[image], return_tensors="pt")

    elif FAMILY == "minicpm":
        msgs = [
            {"role": "user",      "content": prompt_text},
            {"role": "assistant", "content": answer},
        ]
        _, inputs_dict = model.build_conversation_input_ids(
            tokenizer, query=prompt_text, history=[],
            image=image, answer=answer)
        inputs = {k: v.unsqueeze(0).to(model.device) if v.dim() == 0
                  else v.unsqueeze(0).to(model.device)
                  for k, v in inputs_dict.items()}

    else:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ]},
            {"role": "assistant", "content": answer},
        ]
        chat   = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[chat], images=[image],
                           padding=True, return_tensors="pt")

    return {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def generate_answer(prompt_inputs: dict) -> str:
    """Run greedy generation; return decoded answer text."""
    # Strip non-tensor keys before passing to model.generate
    gen_inputs = {k: v for k, v in prompt_inputs.items()
                  if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        if FAMILY == "internvl":
            # InternVL generate is driven by pixel_values + input_ids
            gen_ids = model.generate(
                **gen_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elif FAMILY == "minicpm":
            gen_ids = model.generate(
                **gen_inputs,
                max_new_tokens=5,
                do_sample=False,
            )
        else:
            gen_ids = model.generate(
                **gen_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    prompt_len = prompt_inputs["input_ids"].shape[1]
    new_ids    = gen_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()


def extract_hidden_states(
    full_inputs: dict,
    prompt_len: int,
) -> np.ndarray:
    """
    Run teacher-forcing forward pass on (prompt + answer) and return
    hidden states at the output token position.

    Returns: [num_layers + 1, hidden_dim]
    """
    fwd_inputs = {k: v for k, v in full_inputs.items()
                  if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        out = model(**fwd_inputs, output_hidden_states=True, return_dict=True)

    # hidden_states: tuple of (L+1) tensors, each [1, seq_len, hidden_dim]
    full_len = full_inputs["input_ids"].shape[1]
    n_new    = max(full_len - prompt_len, 1)

    features = []
    for h in out.hidden_states:
        h_seq = h[0]   # [seq_len, hidden_dim]
        if args.pooling == "last":
            vec = h_seq[-1, :].float().cpu().numpy()
        else:
            vec = h_seq[-n_new:, :].float().mean(dim=0).cpu().numpy()
        features.append(vec)

    return np.stack(features)   # [L+1, D]

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

for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Extracting"):

    question_id = str(row["question_id"])
    question    = row["question"]
    gt          = row["answer"].strip().lower()
    split_tag   = "test" if question_id in test_ids else "train_val"

    image = get_pil_image(row["image"])
    if image is None:
        print(f"Warning: no image for question_id={question_id}, skipping.")
        continue

    try:
        # ---- STEP 1: encode prompt ----
        prompt_inputs = encode_prompt(image, question)
        prompt_len    = prompt_inputs["input_ids"].shape[1]

        # ---- STEP 2: generate answer ----
        gen_text = generate_answer(prompt_inputs)

        # ---- STEP 3: EM label ----
        label = 0 if exact_match(gen_text, gt) else 1   # 0=factual, 1=nonfactual

        # ---- STEP 4: teacher-forcing on (prompt + answer) ----
        full_inputs = encode_full(image, question, gen_text)
        features    = extract_hidden_states(full_inputs, prompt_len)

    except Exception as e:
        print(f"Warning: skipping question_id={question_id}: {e}")
        continue

    all_features.append(features)
    all_labels.append(label)
    all_preds.append(gen_text)
    all_gts.append(gt)
    all_ids.append(question_id)
    all_splits.append(split_tag)

# ============================================================
# SAVE
# ============================================================

all_features = np.stack(all_features)
all_labels   = np.array(all_labels)

print(f"\nFeatures shape: {all_features.shape}  (N x layers x hidden_dim)")

unique, counts = np.unique(all_labels, return_counts=True)
total = len(all_labels)
print("Label distribution (0=factual, 1=nonfactual):")
for u, c in zip(unique, counts):
    print(f"  {u}: {c}  ({100*c/total:.1f}%)")

nonfactual = counts[list(unique).index(1)] if 1 in unique else 0
factual    = counts[list(unique).index(0)] if 0 in unique else 0
print(f"\nBalance ratio to reach 50/50: keep all {nonfactual} nonfactual, "
      f"subsample {nonfactual}/{factual} = {nonfactual/max(factual,1):.3f} of factual")

split_arr = np.array(all_splits)
for s in ["test", "train_val"]:
    print(f"  {s}: {(split_arr == s).sum()}")

np.savez(
    OUTPUT_FILE,
    features = all_features,
    labels   = all_labels,
    preds    = np.array(all_preds),
    gts      = np.array(all_gts),
    ids      = np.array(all_ids),
    splits   = split_arr,
)
print(f"\nSaved to {OUTPUT_FILE}")
