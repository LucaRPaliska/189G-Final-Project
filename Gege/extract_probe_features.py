# ============================================================
# VLM Hallucination Probe Pipeline
# Model: Qwen2.5-VL-3B-Instruct
#
# This script:
#   1. Loads POPE-style yes/no benchmark samples
#   2. Runs frozen VLM inference
#   3. Extracts hidden states from ALL layers
#   4. Uses last prompt token representation
#   5. Computes yes/no prediction via logits
#   6. Creates hallucination labels
#   7. Saves probe dataset
#
# Output:
#   eg. probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last.npz
#
# Commands: 
# python3 extract_probe_features.py --model Qwen/Qwen2.5-VL-3B-Instruct --pooling last
# python3 extract_probe_features.py --model Qwen/Qwen2.5-VL-3B-Instruct --pooling mean

# python3 extract_probe_features.py --model OpenGVLab/InternVL2-2B --pooling last
# python3 extract_probe_features.py --model OpenGVLab/InternVL2-2B --pooling mean

# python3 extract_probe_features.py --model microsoft/Phi-3.5-vision-instruct --pooling last
# python3 extract_probe_features.py --model microsoft/Phi-3.5-vision-instruct --pooling mean

# python3 extract_probe_features.py --model deepseek-ai/deepseek-vl-1.3b-base --pooling last
# python3 extract_probe_features.py --model deepseek-ai/deepseek-vl-1.3b-base --pooling mean

# python3 extract_probe_features.py --model openbmb/MiniCPM-V-2 --pooling last
# python3 extract_probe_features.py --model openbmb/MiniCPM-V-2 --pooling mean


# ============================================================

import os
import json
import argparse
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = args.model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16

MAX_SAMPLES = args.max_samples

POOLING = args.pooling

# auto-name output file if not specified
if args.output is not None:
    OUTPUT_FILE = args.output
else:
    model_short = MODEL_NAME.replace("/", "_")
    OUTPUT_FILE = f"probe_dataset_{model_short}_{POOLING}.npz"

# Example:
# data/
#   pope.json
#   images/
#
POPE_JSON_PATH = "data/pope.json"
IMAGE_DIR = "data/images"

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded.")

# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

with open(POPE_JSON_PATH, "r") as f:
    dataset = json.load(f)

if MAX_SAMPLES is not None:
    dataset = dataset[:MAX_SAMPLES]

print(f"Loaded {len(dataset)} samples.")

# ============================================================
# TOKEN IDS
# ============================================================

tokenizer = processor.tokenizer

YES_ID = tokenizer.encode("yes", add_special_tokens=False)[0]
NO_ID = tokenizer.encode("no", add_special_tokens=False)[0]

print("YES token id:", YES_ID)
print("NO token id:", NO_ID)

# ============================================================
# STORAGE
# ============================================================

all_features = []
all_labels = []
all_preds = []
all_gts = []

# ============================================================
# MAIN LOOP
# ============================================================

for sample in tqdm(dataset):

    # --------------------------------------------------------
    # LOAD SAMPLE
    # --------------------------------------------------------

    image_path = os.path.join(
        IMAGE_DIR,
        sample["image"]
    )

    question = sample["text"]
    gt = sample["label"].strip().lower()

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = (
        question
        + " Answer only yes or no."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # --------------------------------------------------------
    # PREDICTION VIA LOGITS
    # --------------------------------------------------------

    # logits shape:
    # [batch, seq_len, vocab]

    logits = outputs.logits

    # final prompt token logits
    final_logits = logits[0, -1]

    yes_logit = final_logits[YES_ID]
    no_logit = final_logits[NO_ID]

    pred = "yes" if yes_logit > no_logit else "no"

    # --------------------------------------------------------
    # HALLUCINATION LABEL
    # --------------------------------------------------------

    hallucination_label = int(pred != gt)

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    # hidden_states:
    # tuple(num_layers + 1)
    #
    # each:
    # [batch, seq_len, hidden_dim]

    hidden_states = outputs.hidden_states

    layer_features = []

    for layer_idx in range(len(hidden_states)):

        h = hidden_states[layer_idx][0]  # [seq_len, hidden_dim]

        if POOLING == "last":
            vec = h[-1, :].float().cpu().numpy()
        else:
            vec = h.float().mean(dim=0).cpu().numpy()

        layer_features.append(vec)

    layer_features = np.stack(layer_features)

    # shape: [num_layers+1, hidden_dim]

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_features.append(layer_features)
    all_labels.append(hallucination_label)
    all_preds.append(pred)
    all_gts.append(gt)

# ============================================================
# SAVE
# ============================================================

all_features = np.stack(all_features)
all_labels = np.array(all_labels)

print(f"Features ({POOLING}) shape:", all_features.shape)
print("Labels shape:", all_labels.shape)

np.savez(
    OUTPUT_FILE,
    features=all_features,
    labels=all_labels,
    preds=np.array(all_preds),
    gts=np.array(all_gts),
)

print(f"Saved to {OUTPUT_FILE}")

# ============================================================
# SHAPES
# ============================================================

# features: [N, num_layers+1, hidden_dim]
# labels:   [N]
#
# Example:
# N=5000, layers=37, hidden_dim=2048 => [5000, 37, 2048]
#
# In train script:
# X = features[:, layer_idx, :]
#
# ============================================================