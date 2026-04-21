"""
Demo 3: Regression — fitting a noisy sine wave.
Showcases MSE loss, Huber loss comparison, and loss landscape.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from neural_viz import (
    NeuralNetwork, Dense, Dropout,
    Adam, MSE,
    Visualizer,
)
from neural_viz.core.losses import HuberLoss


def make_noisy_sine(n=500, noise=0.15, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(X) + rng.randn(n, 1) * noise
    return X.astype(np.float64), y.astype(np.float64)


X, y = make_noisy_sine(n=600, noise=0.2)
split = int(0.8 * len(X))

# Normalize inputs
mu_x, sig_x = X[:split].mean(), X[:split].std()
X_train = (X[:split] - mu_x) / sig_x
X_val   = (X[split:] - mu_x) / sig_x
y_train, y_val = y[:split], y[split:]

model = NeuralNetwork("SineRegressor")
model.add(Dense(64, activation="tanh", init="xavier", name="Dense1"))
model.add(Dense(64, activation="tanh", init="xavier", name="Dense2"))
model.add(Dense(32, activation="tanh", init="xavier", name="Dense3"))
model.add(Dense(1, activation="linear", init="xavier", name="Output"))

model.compile(
    optimizer=Adam(lr=0.003),
    loss=MSE(),
)

_ = model._forward(X_train[:1], training=False)
print(model.get_architecture_summary())

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=True,
    snapshot_interval=5,
)

viz = Visualizer(model, save_dir="demo_regression_output")
viz.show_training()
viz.show_weights()
viz.show_gradient_flow()
viz.show_weight_evolution()
viz.show_activations(X_train[:100])
viz.show_activation_curves()
viz.show_loss_landscape(X_train[:200], y_train[:200], resolution=25, perturbation=0.4)
print("All regression plots saved.")
