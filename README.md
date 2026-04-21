# neural_viz 

A **from-scratch neural network library** built on pure NumPy, designed ground-up for **visualization**. Every layer, weight matrix, gradient, activation, and loss function can be plotted with a single call.

---

## Features

### Core Engine (zero ML libraries)
| Component | What's included |
|---|---|
| **Layers** | `Dense`, `BatchNorm`, `Dropout`, `Flatten`, `Activation` |
| **Activations** | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Linear` |
| **Losses** | `MSE`, `CrossEntropy`, `BinaryCrossEntropy`, `HuberLoss` |
| **Optimizers** | `SGD` (+ Nesterov momentum), `Adam`, `RMSProp` |

### Visualization Suite
| Plot | Description |
|---|---|
| `show_training()` | Train/val loss & accuracy + smoothed batch loss |
| `show_weights()` | Weight + gradient heatmaps for every Dense layer |
| `show_gradient_flow()` | Bar chart of mean/max |∇| per parameter — detects vanishing/exploding gradients |
| `show_activations(X)` | Histograms of post-activation values at every layer |
| `show_activation_curves()` | Mathematical plots of f(x) and f′(x) for each activation used |
| `show_layer_activations(x)` | Per-neuron activation values for a single input |
| `show_decision_boundary(X, y)` | Class regions with confidence shading (2D inputs) |
| `show_loss_landscape(X, y)` | 2D contour + 3D surface of loss along random weight directions |
| `show_weight_evolution()` | Weight and gradient statistics over the entire training run |
| `show_architecture()` | Schematic network graph with neuron circles and connections |

---

## Quick Start

```python
from neural_viz import (
    NeuralNetwork, Dense, Dropout, BatchNorm,
    Adam, CrossEntropy,
    Visualizer,
)
import numpy as np

# Build
model = NeuralNetwork("MyNet")
model.add(Dense(128, activation="relu"))
model.add(BatchNorm())
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer=Adam(lr=0.001), loss=CrossEntropy())

# Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
)

# Visualize everything
viz = Visualizer(model, save_dir="my_output")
viz.plot_all(X_train, y_train)   # saves all 10 plots as PNG

# Or individually
viz.show_training()
viz.show_decision_boundary(X_train, y_train)   # 2D inputs only
viz.show_loss_landscape(X_train, y_train)
```

---

## Installation

```bash
pip install numpy matplotlib
```

No other dependencies. The `neural_viz/` folder is the entire library.

---

## Project Structure

```
neural_viz/
├── __init__.py              # Top-level exports
├── requirements.txt
│
├── core/
│   ├── tensor.py            # Tensor: data + gradients + history
│   ├── layers.py            # Dense, Dropout, BatchNorm, Flatten, Activation
│   ├── activations.py       # ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear
│   ├── losses.py            # MSE, CrossEntropy, BinaryCrossEntropy, HuberLoss
│   ├── optimizers.py        # SGD, Adam, RMSProp
│   └── network.py           # NeuralNetwork (train, predict, snapshot, export)
│
├── viz/
│   ├── plots.py             # Individual plot functions (all return matplotlib Figure)
│   └── dashboard.py         # Visualizer class — unified interface
│
└── examples/
    ├── demo_moons.py        # Binary classification (two moons dataset)
    ├── demo_spiral.py       # 3-class spiral classification
    └── demo_regression.py   # Regression on a noisy sine wave
```

---

## Architecture Details

### Tensor
Every parameter (`W`, `b`, `gamma`, `beta`) is a `Tensor`. It stores:
- `.data` — the NumPy array
- `.grad` — gradient of same shape, zeroed each step
- `._history` — rolling list of `{mean, std, min, max, norm}` snapshots
- `._grad_history` — same for gradients
These histories are what power `show_weight_evolution()`.

### Layer Protocol
Every layer implements:
```python
def forward(self, x, training=True) -> np.ndarray
def backward(self, grad)            -> np.ndarray   # upstream gradient
def get_params(self)                -> List[Tensor]  # trainable tensors
def get_viz_data(self)              -> dict          # full state for visualization
```

### Training Loop
`NeuralNetwork.fit()` runs standard mini-batch SGD:
1. Shuffle dataset
2. For each batch: zero grads → forward → loss → backward → optimizer step
3. Epoch-level metrics collected into `model.history`
4. Full layer snapshots taken every `snapshot_interval` epochs

---

## Examples

### Run the demos
```bash
cd /path/to/parent/of/neural_viz
PYTHONPATH=. python neural_viz/examples/demo_moons.py
PYTHONPATH=. python neural_viz/examples/demo_spiral.py
PYTHONPATH=. python neural_viz/examples/demo_regression.py
```
Each demo trains a network and saves ~10 visualization PNGs to its output folder.

---

## Extending

**Add a new activation:**
```python
# core/activations.py
class Swish(Activation):
    def forward(self, x):
        self._last_input = x
        self._last_output = x / (1 + np.exp(-x))
        return self._last_output

    def backward(self, grad):
        s = 1 / (1 + np.exp(-self._last_input))
        return grad * (self._last_output + s * (1 - self._last_output))
```

**Add a new loss:**
```python
# core/losses.py
class LogCosh(Loss):
    def forward(self, pred, target):
        r = pred - target
        loss = float(np.mean(np.log(np.cosh(r))))
        self._record(loss)
        self._last_pred, self._last_target = pred, target
        return loss

    def backward(self):
        return np.tanh(self._last_pred - self._last_target) / self._last_pred.shape[0]
```

**Custom training callback:**
```python
def lr_schedule(epoch, model):
    if epoch == 20:
        model.optimizer.lr *= 0.1
        print(f"LR dropped to {model.optimizer.lr}")

model.fit(X, y, epochs=50, callbacks=[lr_schedule])
```
