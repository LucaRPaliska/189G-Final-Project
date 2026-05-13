# ============================================================
# Train Hallucination Probe
# ============================================================

'''
Commands: 

python3 train_probe_classifier.py --dataset probe_dataset_Qwen_Qwen2.5-VL-3B-Instruct_last.npz
python3 train_probe_classifier.py --dataset probe_dataset_OpenGVLab_InternVL2-2B_last.npz
python3 train_probe_classifier.py --dataset probe_dataset_microsoft_Phi-3.5-vision-instruct_last.npz
python3 train_probe_classifier.py --dataset probe_dataset_deepseek-ai_deepseek-vl-1.3b-base_last.npz
python3 train_probe_classifier.py --dataset probe_dataset_openbmb_MiniCPM-V-2_last.npz
'''

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
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--layer", type=int, default=20)
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

LAYER_IDX = args.layer
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATA
# ============================================================

data = np.load(args.dataset)

features = data["features"]
labels = data["labels"]

# Select one layer
X = features[:, LAYER_IDX, :]

y = labels

print(X.shape)
print(y.shape)

# ============================================================
# SPLIT
# ============================================================

# First split off 20% for test+val, then split that evenly
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_dataset = torch.utils.data.TensorDataset(
    X_train,
    y_train,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# ============================================================
# MODEL
# ============================================================

hidden_dim = X.shape[1]

model = nn.Sequential(
    nn.Linear(hidden_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
)

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

# ============================================================
# TRAIN
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        logits = model(batch_x).squeeze(-1)

        loss = criterion(logits, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} | Loss = {total_loss:.4f}"
    )

# ============================================================
# EVAL
# ============================================================

model.eval()

with torch.no_grad():

    logits = model(
        X_test.to(DEVICE)
    ).squeeze(-1)

    probs = torch.sigmoid(logits)

    preds = (probs > 0.5).long().cpu().numpy()

acc = accuracy_score(y_test.numpy(), preds)
f1 = f1_score(y_test.numpy(), preds)

print("Accuracy:", acc)
print("F1:", f1)