import numpy as np


class Loss:
    """Abstract loss function."""

    def __init__(self, name="loss"):
        self.name = name
        self._history = []
        self._last_pred = None
        self._last_target = None

    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _record(self, value):
        self._history.append(float(value))
        if len(self._history) > 2000:
            self._history = self._history[-2000:]

    def get_history(self):
        return self._history

    def get_surface_points(self, n=100):
        """
        For 1D regression: returns a 2D grid of loss values
        over (prediction, target) space for visualization.
        """
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error: L = mean((pred - target)^2)"""

    def __init__(self):
        super().__init__("MSE")

    def forward(self, pred, target):
        self._last_pred = pred
        self._last_target = target
        loss = np.mean((pred - target) ** 2)
        self._record(loss)
        return loss

    def backward(self):
        n = self._last_pred.shape[0]
        return 2.0 * (self._last_pred - self._last_target) / n

    def get_surface_points(self, n=60):
        pred = np.linspace(-3, 3, n)
        target = np.linspace(-3, 3, n)
        P, T = np.meshgrid(pred, target)
        L = (P - T) ** 2
        return pred.tolist(), target.tolist(), L.tolist()


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy: L = -mean(y*log(p) + (1-y)*log(1-p))"""

    def __init__(self, eps=1e-12):
        super().__init__("BinaryCrossEntropy")
        self.eps = eps

    def forward(self, pred, target):
        self._last_pred = np.clip(pred, self.eps, 1 - self.eps)
        self._last_target = target
        loss = -np.mean(
            target * np.log(self._last_pred) +
            (1 - target) * np.log(1 - self._last_pred)
        )
        self._record(loss)
        return loss

    def backward(self):
        n = self._last_pred.shape[0]
        p = self._last_pred
        t = self._last_target
        return (-(t / p) + (1 - t) / (1 - p)) / n

    def get_surface_points(self, n=60):
        pred = np.linspace(0.01, 0.99, n)
        target_vals = [0.0, 0.5, 1.0]
        curves = {}
        for t in target_vals:
            curves[t] = (-(t * np.log(pred) + (1 - t) * np.log(1 - pred))).tolist()
        return pred.tolist(), curves


class CrossEntropy(Loss):
    """Categorical Cross Entropy (with softmax outputs): L = -mean(sum(y * log(p)))"""

    def __init__(self, eps=1e-12):
        super().__init__("CrossEntropy")
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: (batch, classes) — softmax outputs
        target: (batch, classes) one-hot OR (batch,) integer labels
        """
        if target.ndim == 1:
            # Convert integer labels to one-hot
            n_classes = pred.shape[1]
            one_hot = np.zeros((target.shape[0], n_classes))
            one_hot[np.arange(target.shape[0]), target.astype(int)] = 1
            target = one_hot

        self._last_pred = np.clip(pred, self.eps, 1.0)
        self._last_target = target
        loss = -np.mean(np.sum(target * np.log(self._last_pred), axis=1))
        self._record(loss)
        return loss

    def backward(self):
        n = self._last_pred.shape[0]
        # Combined softmax + cross-entropy gradient simplifies to:
        return (self._last_pred - self._last_target) / n

    def get_surface_points(self, n=60):
        pred = np.linspace(0.01, 0.99, n)
        loss = -np.log(pred)
        return pred.tolist(), loss.tolist()


class HuberLoss(Loss):
    """Huber loss: quadratic for small errors, linear for large — robust to outliers."""

    def __init__(self, delta=1.0):
        super().__init__(f"Huber(δ={delta})")
        self.delta = delta

    def forward(self, pred, target):
        self._last_pred = pred
        self._last_target = target
        r = pred - target
        abs_r = np.abs(r)
        loss = np.where(abs_r <= self.delta,
                        0.5 * r ** 2,
                        self.delta * (abs_r - 0.5 * self.delta))
        val = float(np.mean(loss))
        self._record(val)
        return val

    def backward(self):
        r = self._last_pred - self._last_target
        n = r.shape[0]
        grad = np.where(np.abs(r) <= self.delta, r, self.delta * np.sign(r))
        return grad / n
