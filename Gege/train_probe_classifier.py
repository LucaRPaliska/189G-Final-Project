# ============================================================
# Train Hallucination Probe
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG
# ============================================================

LAYER_IDX = 20
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATA
# ============================================================

data = np.load("probe_dataset.npz")

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

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