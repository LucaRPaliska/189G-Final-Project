# ============================================================
# Inspect NPZ File
#
# Prints a full summary of any probe dataset .npz file.
#
# Commands:
# python3 inspect_npz.py probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last_em_output.npz
# python3 inspect_npz.py probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last_em_output.npz --samples 10
# ============================================================

import argparse
import numpy as np

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("file",    type=str, help="Path to .npz file")
parser.add_argument("--samples", type=int, default=5,
                    help="Number of example predictions to print (default: 5)")
args = parser.parse_args()

# ============================================================
# LOAD
# ============================================================

data = np.load(args.file, allow_pickle=True)

print("=" * 60)
print(f"File: {args.file}")
print("=" * 60)

# ============================================================
# ARRAYS
# ============================================================

print("\n── Arrays ──────────────────────────────────────────────")

for key in data.files:
    arr = data[key]
    print(f"  {key:<10}  shape={str(arr.shape):<25}  dtype={arr.dtype}")

# ============================================================
# FEATURES
# ============================================================

features = data["features"]
N, L, D  = features.shape

print(f"\n── Features ────────────────────────────────────────────")
print(f"  Samples:    {N}")
print(f"  Layers:     {L}")
print(f"  Hidden dim: {D}")

# ============================================================
# LABELS
# ============================================================

labels = data["labels"]

factual    = (labels == 0).sum()
nonfactual = (labels == 1).sum()

print(f"\n── Labels ──────────────────────────────────────────────")
print(f"  factual    (0): {factual:<6}  ({100 * factual    / len(labels):.1f}%)")
print(f"  nonfactual (1): {nonfactual:<6}  ({100 * nonfactual / len(labels):.1f}%)")
print(f"  model accuracy: {100 * factual / len(labels):.1f}%")

# ============================================================
# SPLITS  (if present)
# ============================================================

if "splits" in data:
    splits = data["splits"]
    print(f"\n── Splits ──────────────────────────────────────────────")
    for s in ["train_val", "test"]:
        mask = splits == s
        n    = mask.sum()
        f    = (labels[mask] == 0).sum()
        nf   = (labels[mask] == 1).sum()
        print(f"  {s:<12}  {n:>5} samples  |  factual={f}  nonfactual={nf}")

# ============================================================
# PREDICTIONS
# ============================================================

if "preds" in data and "gts" in data:
    preds = data["preds"]
    gts   = data["gts"]

    correct   = sum(p.strip().lower().rstrip(".,!?") == g.strip().lower().rstrip(".,!?")
                    for p, g in zip(preds, gts))

    print(f"\n── Predictions ─────────────────────────────────────────")
    print(f"  Correct:   {correct} / {len(preds)}  ({100 * correct / len(preds):.1f}%)")
    print(f"  Incorrect: {len(preds) - correct}")

    print(f"\n  First {args.samples} examples:")
    print(f"  {'#':<5}  {'pred':<8}  {'gold':<8}  {'label':<8}  {'match'}")
    print(f"  {'-'*45}")
    for i in range(min(args.samples, len(preds))):
        match = "✓" if labels[i] == 0 else "✗"
        print(f"  {i:<5}  {preds[i]:<8}  {gts[i]:<8}  {labels[i]:<8}  {match}")

print("\n" + "=" * 60)
