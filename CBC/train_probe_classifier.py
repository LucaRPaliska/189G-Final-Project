# ============================================================
# Train Hallucination Probe
# ============================================================

# Commands:
#
# python3 train_probe_classifier.py --dataset probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last_logit_prompt.npz
# python3 train_probe_classifier.py --dataset probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last_em_output.npz --all_layers
# python3 train_probe_classifier.py --dataset probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last_logit_prompt.npz --early_stopping
# python3 train_probe_classifier.py --dataset probe_dataset_A.npz --cross_test probe_dataset_B.npz

import argparse
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",       type=str, required=True)
parser.add_argument("--layer",         type=int, default=20)
parser.add_argument("--all_layers",    action="store_true",
                    help="Sweep all layers and report best")
parser.add_argument("--early_stopping",action="store_true",
                    help="Stop training when val accuracy stops improving")
parser.add_argument("--patience",      type=int, default=5,
                    help="Early stopping patience in epochs (default: 5)")
parser.add_argument("--balance",       action="store_true",
                    help="Downsample majority class to 50/50 before training")
parser.add_argument("--cross_test",    type=str, default=None,
                    help="Path to a second .npz to evaluate on after training")
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

LAYER_IDX  = args.layer
BATCH_SIZE = 64
EPOCHS     = 50 if args.early_stopping else 10
LR         = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATA
# ============================================================

data = np.load(args.dataset, allow_pickle=True)

features = data["features"]
labels   = data["labels"]

# Use pre-defined splits if present (from --shared_test extraction).
# Fall back to a reproducible random split when absent.
if "splits" in data:
    splits   = data["splits"]
    tv_mask  = splits != "test"
    te_mask  = splits == "test"
    X_tv, y_tv = features[tv_mask], labels[tv_mask]
    X_test = features[te_mask]
    y_test = labels[te_mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.2, random_state=42)

else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        features[:, LAYER_IDX, :], labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ============================================================
# OPTIONAL CROSS-TEST DATASET
# ============================================================

cross_features = None
cross_labels   = None

if args.cross_test is not None:

    cross_data = np.load(args.cross_test, allow_pickle=True)

    if "splits" in cross_data:
        te_mask        = cross_data["splits"] == "test"
        cross_features = cross_data["features"][te_mask]
        cross_labels   = cross_data["labels"][te_mask]
    else:
        cross_features = cross_data["features"]
        cross_labels   = cross_data["labels"]

    print(f"Cross-test samples: {len(cross_features)}")

# ============================================================
# HELPER: BALANCE + SELECT LAYER
# ============================================================

def prepare_layer(layer_idx):
    """
    Select features for one layer, optionally balance classes.
    Returns X_train, X_val, X_test (numpy arrays).
    """

    if "splits" in data:
        Xtr = features[tv_mask,  layer_idx, :]
        Xv  = X_val[:, layer_idx, :] if X_val.ndim == 3 else X_val
        Xte = X_test[:, layer_idx, :] if X_test.ndim == 3 else X_test
    else:
        Xtr = X_train
        Xv  = X_val
        Xte = X_test

    # When splits are pre-defined, re-split train/val by layer
    if "splits" in data:
        Xtr, Xv, ytr, yv = train_test_split(
            features[tv_mask, layer_idx, :], labels[tv_mask],
            test_size=0.2, random_state=42)
        Xte = features[te_mask, layer_idx, :]
    else:
        Xtr = X_train[:, layer_idx, :] if X_train.ndim == 3 else X_train
        Xv  = X_val[:,   layer_idx, :] if X_val.ndim == 3 else X_val
        Xte = X_test[:,  layer_idx, :] if X_test.ndim == 3 else X_test
        ytr, yv = y_train, y_val

    if args.balance:
        pos_idx = np.where(ytr == 1)[0]
        neg_idx = np.where(ytr == 0)[0]
        n_min   = min(len(pos_idx), len(neg_idx))
        rng     = np.random.default_rng(42)
        sel     = np.concatenate([
                      rng.choice(pos_idx, n_min, replace=False),
                      rng.choice(neg_idx, n_min, replace=False),
                  ])
        Xtr = Xtr[sel]
        ytr = ytr[sel]

    return Xtr, ytr, Xv, yv, Xte

# ============================================================
# HELPER: TRAIN ONE PROBE
# ============================================================

def train_probe(layer_idx):
    """Train probe for one layer. Returns (val_acc, test_acc, test_f1)."""

    Xtr, ytr, Xv, yv, Xte = prepare_layer(layer_idx)
    yte = labels[te_mask] if "splits" in data else y_test

    # ---- tensors ----
    X_train_t = torch.tensor(Xtr, dtype=torch.float32)
    X_val_t   = torch.tensor(Xv,  dtype=torch.float32)
    X_test_t  = torch.tensor(Xte, dtype=torch.float32)
    y_train_t = torch.tensor(ytr, dtype=torch.float32)
    y_val_t   = torch.tensor(yv,  dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ---- model ----
    hidden_dim = Xtr.shape[1]

    probe = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    probe     = probe.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)

    # ---- train ----
    best_val_acc = -1.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(EPOCHS):

        probe.train()

        total_loss = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = probe(batch_x).squeeze(-1)
            loss   = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- validation ----
        probe.eval()

        with torch.no_grad():
            val_logits = probe(X_val_t.to(DEVICE)).squeeze(-1)
            val_preds  = (torch.sigmoid(val_logits) > 0.5).long().cpu().numpy()

        val_acc = accuracy_score(yv, val_preds)

        if not args.all_layers:
            print(f"Epoch {epoch+1} | Loss = {total_loss:.4f} | Val Acc = {val_acc:.4f}")

        if args.early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    if not args.all_layers:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    # ---- test eval ----
    probe.load_state_dict(best_state)
    probe.eval()

    with torch.no_grad():
        test_logits = probe(X_test_t.to(DEVICE)).squeeze(-1)
        test_preds  = (torch.sigmoid(test_logits) > 0.5).long().cpu().numpy()

    acc = accuracy_score(yte, test_preds)
    f1  = f1_score(yte, test_preds, zero_division=0)

    # ---- cross-test eval ----
    if cross_features is not None:
        X_cross = torch.tensor(
            cross_features[:, layer_idx, :] if cross_features.ndim == 3
            else cross_features,
            dtype=torch.float32,
        )
        with torch.no_grad():
            c_logits = probe(X_cross.to(DEVICE)).squeeze(-1)
            c_preds  = (torch.sigmoid(c_logits) > 0.5).long().cpu().numpy()
        cross_acc = accuracy_score(cross_labels, c_preds)
        cross_f1  = f1_score(cross_labels, c_preds, zero_division=0)
    else:
        cross_acc = cross_f1 = None

    return best_val_acc, acc, f1, cross_acc, cross_f1

# ============================================================
# RUN
# ============================================================

n_layers = features.shape[1]

if args.all_layers:

    print(f"Sweeping all {n_layers} layers...")

    results = []

    for li in range(n_layers):
        val_acc, test_acc, test_f1, cross_acc, cross_f1 = train_probe(li)
        row = f"  Layer {li:3d} | val={val_acc:.4f} | test_acc={test_acc:.4f} | test_f1={test_f1:.4f}"
        if cross_acc is not None:
            row += f" | cross_acc={cross_acc:.4f}"
        print(row)
        results.append((li, val_acc, test_acc, test_f1))

    best = max(results, key=lambda x: x[1])
    print(f"\nBest layer: {best[0]}  val={best[1]:.4f}  test_acc={best[2]:.4f}  test_f1={best[3]:.4f}")

else:

    val_acc, acc, f1, cross_acc, cross_f1 = train_probe(LAYER_IDX)

    print("Accuracy:", acc)
    print("F1:", f1)

    if cross_acc is not None:
        print("Cross-test Accuracy:", cross_acc)
        print("Cross-test F1:", cross_f1)
