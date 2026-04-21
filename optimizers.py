import numpy as np


class Optimizer:
    """Base optimizer class."""

    def __init__(self, lr, name="optimizer"):
        self.lr = lr
        self.name = name
        self._step = 0
        self._lr_history = []
        self._update_norm_history = []  # norm of parameter updates per step

    def step(self, params):
        self._step += 1
        self._lr_history.append(self.lr)

    def get_viz_data(self):
        return {
            "name": self.name,
            "lr": self.lr,
            "step": self._step,
            "lr_history": self._lr_history[-200:],
            "update_norm_history": self._update_norm_history[-200:],
        }


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and weight decay.
    Update: v = μv - lr*grad; param += v
    """

    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(lr, name=f"SGD(lr={lr}, momentum={momentum})")
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._velocities = {}

    def step(self, params):
        super().step(params)
        total_update_norm = 0.0

        for param in params:
            if not param.requires_grad or param.grad is None:
                continue

            pid = id(param)
            if pid not in self._velocities:
                self._velocities[pid] = np.zeros_like(param.data)

            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            v = self._velocities[pid]
            v = self.momentum * v - self.lr * grad
            self._velocities[pid] = v

            if self.nesterov:
                update = self.momentum * v - self.lr * grad
            else:
                update = v

            total_update_norm += np.linalg.norm(update)
            param.data += update

        self._update_norm_history.append(total_update_norm)
        if len(self._update_norm_history) > 500:
            self._update_norm_history = self._update_norm_history[-500:]


class Adam(Optimizer):
    """
    Adam optimizer: adaptive learning rates using first and second moment estimates.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(lr, name=f"Adam(lr={lr})")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}   # first moments
        self._v = {}   # second moments

    def step(self, params):
        super().step(params)
        total_update_norm = 0.0
        t = self._step

        for param in params:
            if not param.requires_grad or param.grad is None:
                continue

            pid = id(param)
            if pid not in self._m:
                self._m[pid] = np.zeros_like(param.data)
                self._v[pid] = np.zeros_like(param.data)

            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            m = self._m[pid]
            v = self._v[pid]

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2

            self._m[pid] = m
            self._v[pid] = v

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            total_update_norm += np.linalg.norm(update)
            param.data += update

        self._update_norm_history.append(total_update_norm)
        if len(self._update_norm_history) > 500:
            self._update_norm_history = self._update_norm_history[-500:]

    def get_moment_stats(self, params):
        """Return moment statistics for visualization."""
        stats = []
        for param in params:
            pid = id(param)
            if pid in self._m:
                stats.append({
                    "name": param.name,
                    "m_norm": float(np.linalg.norm(self._m[pid])),
                    "v_norm": float(np.linalg.norm(self._v[pid])),
                    "effective_lr": float(np.mean(
                        self.lr / (np.sqrt(self._v[pid] / (1 - self.beta2 ** max(self._step, 1))) + self.eps)
                    )),
                })
        return stats


class RMSProp(Optimizer):
    """
    RMSProp: maintains a moving average of squared gradients for adaptive step sizes.
    """

    def __init__(self, lr=0.001, rho=0.9, eps=1e-8, weight_decay=0.0):
        super().__init__(lr, name=f"RMSProp(lr={lr})")
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self._sq_avg = {}

    def step(self, params):
        super().step(params)
        total_update_norm = 0.0

        for param in params:
            if not param.requires_grad or param.grad is None:
                continue

            pid = id(param)
            if pid not in self._sq_avg:
                self._sq_avg[pid] = np.zeros_like(param.data)

            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            sq = self._sq_avg[pid]
            sq = self.rho * sq + (1 - self.rho) * grad ** 2
            self._sq_avg[pid] = sq

            update = -self.lr * grad / (np.sqrt(sq) + self.eps)
            total_update_norm += np.linalg.norm(update)
            param.data += update

        self._update_norm_history.append(total_update_norm)
        if len(self._update_norm_history) > 500:
            self._update_norm_history = self._update_norm_history[-500:]
