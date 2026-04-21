import numpy as np


class Tensor:
    """
    Core data structure wrapping a NumPy array.
    Tracks values, gradients, and metadata for visualization.
    """

    def __init__(self, data, name=None, requires_grad=True):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.name = name or "tensor"
        self.requires_grad = requires_grad

        # Visualization metadata
        self._history = []          # rolling history of mean values
        self._grad_history = []     # rolling history of mean grad norms
        self._update_count = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        t = Tensor(self.data.T, name=self.name + ".T")
        t.grad = self.grad.T
        return t

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def record_stats(self):
        """Called after each update to log stats for visualization."""
        self._history.append({
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "norm": float(np.linalg.norm(self.data)),
        })
        if self.grad is not None and np.any(self.grad != 0):
            self._grad_history.append({
                "mean": float(np.mean(self.grad)),
                "std": float(np.std(self.grad)),
                "norm": float(np.linalg.norm(self.grad)),
                "max_abs": float(np.max(np.abs(self.grad))),
            })
        self._update_count += 1
        # Keep only last 500 entries
        if len(self._history) > 500:
            self._history = self._history[-500:]
        if len(self._grad_history) > 500:
            self._grad_history = self._grad_history[-500:]

    def get_stats(self):
        return {
            "shape": self.data.shape,
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "grad_norm": float(np.linalg.norm(self.grad)) if self.grad is not None else 0.0,
            "update_count": self._update_count,
        }

    def __repr__(self):
        return f"Tensor(name={self.name!r}, shape={self.shape}, mean={np.mean(self.data):.4f})"
