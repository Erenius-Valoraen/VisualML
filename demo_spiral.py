"""
Demo 2: Multi-class spiral classification.
A harder problem that tests deeper networks and shows cleaner gradient flow.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from neural_viz import (
    NeuralNetwork, Dense, Dropout, BatchNorm,
    Adam, CrossEntropy,
    Visualizer,
)


def make_spirals(n_per_class=200, n_classes=3, seed=0):
    """Andrej Karpathy's spiral dataset."""
    rng = np.random.RandomState(seed)
    N, C = n_per_class, n_classes
    X = np.zeros((N * C, 2))
    y = np.zeros(N * C, dtype=int)
    for c in range(C):
        ix = range(N * c, N * (c + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(c * 4, (c + 1) * 4, N) + rng.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = c
    idx = rng.permutation(N * C)
    return X[idx].astype(np.float64), y[idx]


X, y = make_spirals(n_per_class=300, n_classes=3)
split = int(0.8 * len(X))

mu, sig = X[:split].mean(0), X[:split].std(0) + 1e-8
X_train = (X[:split] - mu) / sig
X_val   = (X[split:] - mu) / sig
y_train, y_val = y[:split], y[split:]

model = NeuralNetwork("SpiralClassifier")
model.add(Dense(128, activation="relu", name="Dense1"))
model.add(BatchNorm(name="BN1"))
model.add(Dense(128, activation="relu", name="Dense2"))
model.add(BatchNorm(name="BN2"))
model.add(Dropout(0.25, name="Drop1"))
model.add(Dense(64, activation="relu", name="Dense3"))
model.add(Dense(3, activation="softmax", name="Output"))

model.compile(
    optimizer=Adam(lr=0.002),
    loss=CrossEntropy(),
)

_ = model._forward(X_train[:1], training=False)
print(model.get_architecture_summary())

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=True,
    snapshot_interval=5,
)

viz = Visualizer(model, save_dir="demo_spiral_output")
viz.plot_all(X_train, y_train)
print("\nFinal val accuracy:", model.history["val_acc"][-1])
