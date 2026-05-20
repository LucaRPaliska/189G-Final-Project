"""
train_probe.py

Train a linear hallucination probe on VLM hidden states collected by
collect_hidden_states.py.

Features:
  - Respects pre-defined train_val / test splits stored in the npz
  - Balances training data (equal factual / nonfactual)
  - Cross-entropy loss (BCEWithLogitsLoss)
  - Early stopping on validation accuracy
  - Optional cross-model test: evaluate a probe trained on model A on model B's test set

Usage:
  python train_probe.py --dataset hidden_states_Qwen_Qwen2.5-VL-3B-Instruct_all_last.npz
  python train_probe.py --dataset hidden_states_A.npz --layer 20
  python train_probe.py --dataset hidden_states_A.npz --cross_test hidden_states_B.npz
  python train_probe.py --dataset hidden_states_A.npz --all_layers
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    type=str, required=True,
                    help="Path to .npz produced by collect_hidden_states.py")
parser.add_argument("--layer",      type=int, default=20,
                    help="Which layer's hidden state to use as features")
parser.add_argument("--all_layers", action="store_true",
                    help="Sweep all layers and report best; overrides --layer")
parser.add_argument("--cross_test", type=str, default=None,
                    help="Optional: path to another model's .npz for cross-model eval")
parser.add_argument("--batch_size", type=int,   default=64)
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--patience",   type=int,   default=5,
                    help="Early stopping patience (epochs without val acc improvement)")
parser.add_argument("--hidden",     type=int,   default=512,
                    help="Hidden dim of the 1-hidden-layer MLP probe")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATA
# ============================================================

data   = np.load(args.dataset, allow_pickle=True)
feats  = data["features"]   # [N, num_layers+1, hidden_dim]
labels = data["labels"]     # [N]

# "splits" key is present in Luca's npz; Gege's npz omits it.
# Fall back to a reproducible 80/10/10 random split when absent.
if "splits" in data:
    splits = data["splits"]
else:
    print("Note: no 'splits' key found (Gege-format npz). Using random 80/10/10 split.")
    N   = len(labels)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    n_test = max(1, int(N * 0.1))
    n_val  = max(1, int(N * 0.1))
    splits = np.array(["train_val"] * N)
    splits[idx[:n_test]]            = "test"
    splits[idx[n_test:n_test+n_val]]= "val"   # treated as train_val below

n_layers   = feats.shape[1]
hidden_dim = feats.shape[2]
print(f"Loaded: {feats.shape[0]} samples | {n_layers} layers | dim={hidden_dim}")

# ---- Masks ----
tv_mask   = splits != "test"
test_mask = splits == "test"

# Optional cross-model test set
cross_feats  = None
cross_labels = None
if args.cross_test is not None:
    ct = np.load(args.cross_test, allow_pickle=True)
    cross_feats  = ct["features"]
    cross_labels = ct["labels"]
    cross_mask   = ct["splits"] == "test"
    cross_feats  = cross_feats[cross_mask]
    cross_labels = cross_labels[cross_mask]
    print(f"Cross-model test set: {cross_feats.shape[0]} samples")

# ============================================================
# BALANCE HELPER
# ============================================================

def balance(X, y, seed=42):
    """Downsample majority class to match minority class size."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_min   = min(len(pos_idx), len(neg_idx))
    rng     = np.random.default_rng(seed)
    sel_pos = rng.choice(pos_idx, size=n_min, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_min, replace=False)
    idx     = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]

# ============================================================
# TRAIN + EVAL FOR ONE LAYER
# ============================================================

def run_layer(layer_idx: int) -> dict:
    """Train probe on one layer; return metrics dict."""

    X_tv   = feats[tv_mask,   layer_idx, :]
    y_tv   = labels[tv_mask]
    X_test = feats[test_mask, layer_idx, :]
    y_test = labels[test_mask]

    # Balance then split 80 / 20 train / val
    X_bal, y_bal = balance(X_tv, y_tv)
    n_val        = max(1, int(len(X_bal) * 0.2))
    X_val_np,   y_val_np   = X_bal[:n_val],  y_bal[:n_val]
    X_train_np, y_train_np = X_bal[n_val:],  y_bal[n_val:]

    def t(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    X_train_t, y_train_t = t(X_train_np, y_train_np)
    X_val_t,   y_val_t   = t(X_val_np,   y_val_np)
    X_test_t,  y_test_t  = t(X_test,     y_test)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True,
    )

    # ---- Model ----
    probe = nn.Sequential(
        nn.Linear(hidden_dim, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, 1),
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    # ---- Train with early stopping ----
    best_val_acc = -1.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):

        probe.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by  = bx.to(DEVICE), by.to(DEVICE)
            logits  = probe(bx).squeeze(-1)
            loss    = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        probe.eval()
        with torch.no_grad():
            v_logits = probe(X_val_t.to(DEVICE)).squeeze(-1)
            v_preds  = (torch.sigmoid(v_logits) > 0.5).long().cpu().numpy()
        val_acc = accuracy_score(y_val_np, v_preds)

        if not args.all_layers:
            print(f"  Epoch {epoch:3d} | loss={total_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                if not args.all_layers:
                    print(f"  Early stop at epoch {epoch} (best val={best_val_acc:.4f})")
                break

    # ---- Test evaluation ----
    probe.load_state_dict(best_state)
    probe.eval()

    with torch.no_grad():
        t_logits = probe(X_test_t.to(DEVICE)).squeeze(-1)
        t_preds  = (torch.sigmoid(t_logits) > 0.5).long().cpu().numpy()

    acc = accuracy_score(y_test, t_preds)
    f1  = f1_score(y_test, t_preds, zero_division=0)

    result = {"layer": layer_idx, "val_acc": best_val_acc, "test_acc": acc, "test_f1": f1}

    # Optional cross-model eval
    if cross_feats is not None:
        X_c = torch.tensor(cross_feats[:, layer_idx, :], dtype=torch.float32)
        with torch.no_grad():
            c_logits = probe(X_c.to(DEVICE)).squeeze(-1)
            c_preds  = (torch.sigmoid(c_logits) > 0.5).long().cpu().numpy()
        result["cross_acc"] = accuracy_score(cross_labels, c_preds)
        result["cross_f1"]  = f1_score(cross_labels, c_preds, zero_division=0)

    return result

# ============================================================
# RUN
# ============================================================

if args.all_layers:
    print(f"Sweeping all {n_layers} layers ...")
    results = []
    for li in range(n_layers):
        r = run_layer(li)
        results.append(r)
        print(
            f"  Layer {li:3d} | val={r['val_acc']:.4f} | "
            f"test_acc={r['test_acc']:.4f} | test_f1={r['test_f1']:.4f}"
            + (f" | cross_acc={r['cross_acc']:.4f}" if "cross_acc" in r else "")
        )

    best = max(results, key=lambda x: x["val_acc"])
    print(f"\nBest layer: {best['layer']}  val={best['val_acc']:.4f}  "
          f"test_acc={best['test_acc']:.4f}  test_f1={best['test_f1']:.4f}")

else:
    print(f"Training probe on layer {args.layer} ...")
    print(f"Train/Val/Test sizes (after balance):")

    X_tv = feats[tv_mask, args.layer, :]
    y_tv = labels[tv_mask]
    X_bal, y_bal = balance(X_tv, y_tv)
    n_val = max(1, int(len(X_bal) * 0.2))
    print(f"  Train: {len(X_bal) - n_val} | Val: {n_val} | Test: {test_mask.sum()}")

    r = run_layer(args.layer)
    print(f"\n[Test] Accuracy={r['test_acc']:.4f} | F1={r['test_f1']:.4f}")
    if "cross_acc" in r:
        print(f"[Cross-model Test] Accuracy={r['cross_acc']:.4f} | F1={r['cross_f1']:.4f}")
