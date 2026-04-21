import numpy as np


class Activation:
    """Base class for activation functions."""

    def __init__(self):
        self._last_input = None
        self._last_output = None
        self.name = "activation"

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def get_curve_points(self, x_range=(-5, 5), n=200):
        """Returns (x, y) for plotting the activation curve."""
        x = np.linspace(x_range[0], x_range[1], n)
        y = self.forward(x.reshape(-1, 1)).ravel()
        return x, y

    def get_derivative_points(self, x_range=(-5, 5), n=200):
        """Returns (x, dy/dx) for plotting the derivative."""
        x = np.linspace(x_range[0], x_range[1], n)
        inp = x.reshape(-1, 1)
        _ = self.forward(inp)
        g = self.backward(np.ones_like(inp))
        return x, g.ravel()


class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.name = "ReLU"

    def forward(self, x):
        self._last_input = x
        self._last_output = np.maximum(0, x)
        return self._last_output

    def backward(self, grad):
        return grad * (self._last_input > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.name = f"LeakyReLU(α={alpha})"

    def forward(self, x):
        self._last_input = x
        self._last_output = np.where(x > 0, x, self.alpha * x)
        return self._last_output

    def backward(self, grad):
        dx = np.where(self._last_input > 0, 1.0, self.alpha)
        return grad * dx


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"

    def forward(self, x):
        self._last_output = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._last_output

    def backward(self, grad):
        s = self._last_output
        return grad * s * (1 - s)


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.name = "Tanh"

    def forward(self, x):
        self._last_output = np.tanh(x)
        return self._last_output

    def backward(self, grad):
        return grad * (1 - self._last_output ** 2)


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"

    def forward(self, x):
        # Numerically stable softmax
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self._last_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self._last_output

    def backward(self, grad):
        # Simplified: assumes grad already accounts for softmax-cross-entropy combo
        return grad


class Linear(Activation):
    def __init__(self):
        super().__init__()
        self.name = "Linear"

    def forward(self, x):
        self._last_input = x
        self._last_output = x
        return x

    def backward(self, grad):
        return grad
