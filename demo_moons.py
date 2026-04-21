"""
Demo 1: Binary classification on the classic "two moons" dataset.
Showcases decision boundary, training curves, and all visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from neural_viz import (
    NeuralNetwork, Dense, Dropout, BatchNorm,
    Adam, BinaryCrossEntropy, Sigmoid,
    Visualizer,
)


# ─────────────────────────────────────────────────────────────
# Data: two interleaved half-circles
# ─────────────────────────────────────────────────────────────

def make_moons(n=600, noise=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n_each = n // 2
    t = np.linspace(0, np.pi, n_each)
    X1 = np.c_[np.cos(t), np.sin(t)] + rng.randn(n_each, 2) * noise
    X2 = np.c_[1 - np.cos(t), -np.sin(t) + 0.5] + rng.randn(n_each, 2) * noise
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)]).reshape(-1, 1)
    idx = rng.permutation(n)
    return X[idx].astype(np.float64), y[idx].astype(np.float64)


def normalize(X_train, X_val):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma


# ─────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────

X, y = make_moons(n=800, noise=0.2)
split = int(0.8 * len(X))
X_train, X_val = normalize(X[:split], X[split:])
y_train, y_val = y[:split], y[split:]

model = NeuralNetwork("MoonsClassifier")
model.add(Dense(64, activation="relu", init="he", name="Dense1"))
model.add(BatchNorm(name="BN1"))
model.add(Dropout(0.2, name="Drop1"))
model.add(Dense(64, activation="relu", init="he", name="Dense2"))
model.add(Dense(32, activation="relu", init="he", name="Dense3"))
model.add(Dense(1, activation="sigmoid", init="xavier", name="Output"))

model.compile(
    optimizer=Adam(lr=0.003, weight_decay=1e-4),
    loss=BinaryCrossEntropy(),
)

# Trigger build with a dummy forward pass
_ = model._forward(X_train[:1], training=False)
print(model.get_architecture_summary())

# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=True,
    snapshot_interval=5,
)

# ─────────────────────────────────────────────────────────────
# Visualize
# ─────────────────────────────────────────────────────────────

viz = Visualizer(model, save_dir="demo_moons_output")
viz.plot_all(X_train, y_train)
print("\nFinal val accuracy:", model.history["val_acc"][-1])
