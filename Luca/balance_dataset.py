"""
balance_dataset.py

Reads one or more .npz files produced by collect_hidden_states.py,
reports the factual/nonfactual ratio for each model, and saves a
50/50-balanced version to Luca/balanced/<original_filename>.npz.

Only the train_val portion is balanced. The test split is kept as-is
(shared across models; already ~50/50 by construction).

Usage:
  # Single model
  python balance_dataset.py hidden_states_Qwen_Qwen2.5-VL-3B-Instruct_all_last.npz

  # All models at once
  python balance_dataset.py hidden_states_*.npz

  # Custom output dir
  python balance_dataset.py hidden_states_*.npz --out_dir my_balanced/
"""

import os
import sys
import argparse
import numpy as np

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="+",
                    help="Path(s) to .npz file(s) from collect_hidden_states.py")
parser.add_argument(
    "--out_dir", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "balanced"),
    help="Output directory for balanced .npz files (default: Luca/balanced/)",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ============================================================
# RATIO TABLE (printed at the end)
# ============================================================

summary_rows = []

# ============================================================
# PROCESS EACH FILE
# ============================================================

for npz_path in args.datasets:

    if not os.path.exists(npz_path):
        print(f"[SKIP] File not found: {npz_path}")
        continue

    fname = os.path.basename(npz_path)
    print(f"\n{'='*60}")
    print(f"Model file: {fname}")
    print(f"{'='*60}")

    data   = np.load(npz_path, allow_pickle=True)
    feats  = data["features"]   # [N, L, D]
    labels = data["labels"]     # [N]
    splits = data["splits"]     # [N]

    # ---- Report overall distribution ----
    tv_mask   = splits == "train_val"
    test_mask = splits == "test"

    def report(mask, name):
        y = labels[mask]
        total = len(y)
        if total == 0:
            print(f"  {name}: 0 samples")
            return
        factual    = (y == 0).sum()
        nonfactual = (y == 1).sum()
        acc        = factual / total
        print(f"  {name} ({total} samples):")
        print(f"    factual   (label=0): {factual:5d}  ({100*factual/total:.1f}%)")
        print(f"    nonfactual(label=1): {nonfactual:5d}  ({100*nonfactual/total:.1f}%)")
        print(f"    model accuracy (EM): {100*acc:.1f}%")
        return factual, nonfactual, acc

    tv_stats   = report(tv_mask,   "train_val")
    test_stats = report(test_mask, "test     ")

    if tv_stats is None:
        print("  No train_val samples — skipping.")
        continue

    tv_factual, tv_nonfactual, acc = tv_stats

    # ---- Balance train_val: downsample majority class ----
    rng     = np.random.default_rng(args.seed)
    n_min   = min(tv_factual, tv_nonfactual)

    if n_min == 0:
        print("  WARNING: one class is empty; cannot balance. Saving as-is.")
        bal_tv_idx = np.where(tv_mask)[0]
    else:
        tv_idx   = np.where(tv_mask)[0]
        pos_idx  = tv_idx[labels[tv_idx] == 1]   # nonfactual
        neg_idx  = tv_idx[labels[tv_idx] == 0]   # factual
        sel_pos  = rng.choice(pos_idx, size=n_min, replace=False)
        sel_neg  = rng.choice(neg_idx, size=n_min, replace=False)
        bal_tv_idx = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(bal_tv_idx)

    test_idx = np.where(test_mask)[0]
    keep_idx = np.concatenate([bal_tv_idx, test_idx])
    keep_idx = np.sort(keep_idx)

    # ---- Report what we kept ----
    n_kept = len(bal_tv_idx)
    ratio_kept = tv_nonfactual / max(tv_factual, 1)
    print(f"\n  Balancing train_val:")
    print(f"    keeping {n_min} factual + {n_min} nonfactual = {n_kept} total")
    print(f"    discarded {max(tv_factual, tv_nonfactual) - n_min} majority-class samples")
    print(f"    subsample ratio applied: {ratio_kept:.4f}  "
          f"({'factual' if tv_factual > tv_nonfactual else 'nonfactual'} was majority)")

    # ---- Save ----
    out_path = os.path.join(args.out_dir, fname)

    # Re-build arrays with only kept indices
    saved = {}
    for key in data.files:
        arr = data[key]
        if arr.shape[0] == feats.shape[0]:   # N-length array
            saved[key] = arr[keep_idx]
        else:
            saved[key] = arr

    # Overwrite splits for balanced train_val indices
    new_splits = saved["splits"].copy()
    # All bal_tv_idx rows are "train_val" — no change needed; just sanity check
    assert (new_splits[: len(bal_tv_idx)] == "train_val").all(), \
        "Unexpected split values after reindex"

    np.savez(out_path, **saved)
    print(f"  Saved balanced file → {out_path}")

    # ---- Collect summary row ----
    # Infer model name from filename
    model_name = fname.replace("hidden_states_", "").replace("_all_last.npz", "") \
                       .replace("_all_mean.npz", "").replace(".npz", "")
    summary_rows.append({
        "model":          model_name,
        "total_samples":  len(labels),
        "model_acc_em":   f"{100*acc:.1f}%",
        "factual":        tv_factual,
        "nonfactual":     tv_nonfactual,
        "ratio_needed":   f"{ratio_kept:.4f}",
        "majority":       "factual" if tv_factual > tv_nonfactual else "nonfactual",
        "balanced_per_class": n_min,
    })

# ============================================================
# SUMMARY TABLE
# ============================================================

if summary_rows:
    print(f"\n{'='*80}")
    print("RATIO SUMMARY  (how much to keep from each class to reach 50/50)")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Acc(EM)':>8} {'Factual':>8} {'NonFact':>8} "
          f"{'Ratio':>8} {'Majority':>10} {'Bal/cls':>8}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['model']:<45} {r['model_acc_em']:>8} "
              f"{r['factual']:>8} {r['nonfactual']:>8} "
              f"{r['ratio_needed']:>8} {r['majority']:>10} "
              f"{r['balanced_per_class']:>8}")
    print(f"{'='*80}")
    print("Ratio = nonfactual / factual  (fraction of factual samples to keep)")
    print("Balanced files saved to:", args.out_dir)
