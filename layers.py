import numpy as np
from .tensor import Tensor
from .activations import Linear, ReLU, Sigmoid, Tanh, Softmax, LeakyReLU


def _get_activation(name):
    mapping = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax,
        "linear": Linear,
        "none": Linear,
        "leaky_relu": LeakyReLU,
    }
    if name is None or name.lower() not in mapping:
        return Linear()
    return mapping[name.lower()]()


class Layer:
    """Abstract base layer."""

    def __init__(self, name="layer"):
        self.name = name
        self.trainable = True
        self._built = False
        self._input_cache = None
        self._output_cache = None
        self._input_shape = None
        self._output_shape = None
        # Activation stats per forward pass for visualization
        self._activation_history = []

    def build(self, input_shape):
        pass

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def get_params(self):
        """Returns list of Tensor parameters."""
        return []

    def get_grads(self):
        """Returns list of gradient arrays."""
        return []

    def get_viz_data(self):
        """Returns dict of visualization data for this layer."""
        return {"name": self.name, "type": self.__class__.__name__}


class Dense(Layer):
    """
    Fully-connected layer: y = xW + b
    Supports He/Xavier initialization, optional activation.
    """

    def __init__(self, units, activation=None, use_bias=True,
                 init="he", name=None):
        super().__init__(name=name or f"Dense({units})")
        self.units = units
        self.use_bias = use_bias
        self.init = init
        self.activation = _get_activation(activation) if isinstance(activation, str) else (activation or Linear())
        self.W = None
        self.b = None
        self._pre_activation = None

    def build(self, input_shape):
        fan_in = input_shape[-1]
        fan_out = self.units

        if self.init == "he":
            std = np.sqrt(2.0 / fan_in)
        elif self.init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            std = 0.01

        self.W = Tensor(
            np.random.randn(fan_in, fan_out) * std,
            name=f"{self.name}.W"
        )
        if self.use_bias:
            self.b = Tensor(
                np.zeros((1, fan_out)),
                name=f"{self.name}.b"
            )
        self._built = True
        self._input_shape = input_shape
        self._output_shape = (input_shape[0], fan_out)

    def forward(self, x, training=True):
        if not self._built:
            self.build(x.shape)

        self._input_cache = x
        self._pre_activation = x @ self.W.data
        if self.use_bias:
            self._pre_activation += self.b.data

        out = self.activation.forward(self._pre_activation)
        self._output_cache = out

        # Track activation stats
        self._activation_history.append({
            "mean": float(np.mean(out)),
            "std": float(np.std(out)),
            "dead": float(np.mean(out == 0)) if hasattr(self.activation, 'name') and 'ReLU' in self.activation.name else 0.0,
        })
        if len(self._activation_history) > 500:
            self._activation_history = self._activation_history[-500:]

        return out

    def backward(self, grad):
        # Backprop through activation
        d_pre = self.activation.backward(grad)

        # Gradients for W and b
        batch_size = self._input_cache.shape[0]
        self.W.grad = (self._input_cache.T @ d_pre) / batch_size
        if self.use_bias:
            self.b.grad = np.mean(d_pre, axis=0, keepdims=True)

        # Record stats for visualization
        self.W.record_stats()
        if self.use_bias:
            self.b.record_stats()

        # Pass gradient upstream
        return d_pre @ self.W.data.T

    def get_params(self):
        params = [self.W]
        if self.use_bias:
            params.append(self.b)
        return params

    def get_viz_data(self):
        data = {
            "name": self.name,
            "type": "Dense",
            "units": self.units,
            "activation": self.activation.name,
            "activation_history": self._activation_history[-100:],
        }
        if self.W is not None:
            data["W_shape"] = self.W.shape
            data["W_data"] = self.W.data.tolist()
            data["W_grad"] = self.W.grad.tolist()
            data["W_stats"] = self.W.get_stats()
            data["W_history"] = self.W._history[-100:]
            data["W_grad_history"] = self.W._grad_history[-100:]
        if self.b is not None:
            data["b_data"] = self.b.data.tolist()
            data["b_stats"] = self.b.get_stats()
        return data


class Dropout(Layer):
    """
    Dropout regularization — zeros out random neurons during training.
    """

    def __init__(self, rate=0.5, name=None):
        super().__init__(name=name or f"Dropout({rate})")
        self.rate = rate
        self._mask = None
        self.trainable = False

    def build(self, input_shape):
        self._built = True
        self._input_shape = input_shape
        self._output_shape = input_shape

    def forward(self, x, training=True):
        if not self._built:
            self.build(x.shape)

        if training:
            self._mask = (np.random.rand(*x.shape) > self.rate).astype(float)
            out = x * self._mask / (1.0 - self.rate)
        else:
            self._mask = None
            out = x

        self._input_cache = x
        self._output_cache = out
        return out

    def backward(self, grad):
        if self._mask is not None:
            return grad * self._mask / (1.0 - self.rate)
        return grad

    def get_viz_data(self):
        return {
            "name": self.name,
            "type": "Dropout",
            "rate": self.rate,
            "mask_sparsity": float(np.mean(self._mask == 0)) if self._mask is not None else self.rate,
        }


class BatchNorm(Layer):
    """
    Batch Normalization layer.
    Normalizes inputs to zero mean, unit variance, then scales/shifts.
    """

    def __init__(self, momentum=0.9, eps=1e-5, name=None):
        super().__init__(name=name or "BatchNorm")
        self.momentum = momentum
        self.eps = eps
        self.gamma = None
        self.beta = None
        self._running_mean = None
        self._running_var = None
        self._norm_cache = None

    def build(self, input_shape):
        d = input_shape[-1]
        self.gamma = Tensor(np.ones((1, d)), name=f"{self.name}.gamma")
        self.beta = Tensor(np.zeros((1, d)), name=f"{self.name}.beta")
        self._running_mean = np.zeros((1, d))
        self._running_var = np.ones((1, d))
        self._built = True
        self._input_shape = input_shape
        self._output_shape = input_shape

    def forward(self, x, training=True):
        if not self._built:
            self.build(x.shape)

        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            self._running_mean = self.momentum * self._running_mean + (1 - self.momentum) * mean
            self._running_var = self.momentum * self._running_var + (1 - self.momentum) * var
        else:
            mean = self._running_mean
            var = self._running_var

        x_hat = (x - mean) / np.sqrt(var + self.eps)
        self._norm_cache = (x, x_hat, mean, var)
        out = self.gamma.data * x_hat + self.beta.data
        self._output_cache = out
        return out

    def backward(self, grad):
        x, x_hat, mean, var = self._norm_cache
        N = x.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.gamma.grad = np.sum(grad * x_hat, axis=0, keepdims=True)
        self.beta.grad = np.sum(grad, axis=0, keepdims=True)

        dx_hat = grad * self.gamma.data
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std_inv ** 3, axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=0, keepdims=True)
        dx = dx_hat * std_inv + dvar * 2.0 * (x - mean) / N + dmean / N

        self.gamma.record_stats()
        self.beta.record_stats()
        return dx

    def get_params(self):
        return [self.gamma, self.beta]

    def get_viz_data(self):
        data = {
            "name": self.name,
            "type": "BatchNorm",
            "momentum": self.momentum,
            "eps": self.eps,
        }
        if self.gamma is not None:
            data["gamma"] = self.gamma.data.tolist()
            data["beta"] = self.beta.data.tolist()
            data["running_mean"] = self._running_mean.tolist()
            data["running_var"] = self._running_var.tolist()
        return data


class Flatten(Layer):
    """Flattens input (batch, ...) → (batch, product of rest)."""

    def __init__(self, name=None):
        super().__init__(name=name or "Flatten")
        self._original_shape = None
        self.trainable = False

    def build(self, input_shape):
        self._built = True
        self._input_shape = input_shape

    def forward(self, x, training=True):
        if not self._built:
            self.build(x.shape)
        self._original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._original_shape)

    def get_viz_data(self):
        return {
            "name": self.name,
            "type": "Flatten",
            "input_shape": list(self._original_shape) if self._original_shape else None,
        }


class Activation(Layer):
    """Standalone activation layer (wraps an activation function)."""

    def __init__(self, activation, name=None):
        from .activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear
        super().__init__(name=name or str(activation))
        if isinstance(activation, str):
            self.act = _get_activation(activation)
        else:
            self.act = activation
        self.name = name or self.act.name
        self.trainable = False

    def build(self, input_shape):
        self._built = True
        self._input_shape = input_shape
        self._output_shape = input_shape

    def forward(self, x, training=True):
        if not self._built:
            self.build(x.shape)
        out = self.act.forward(x)
        self._output_cache = out
        return out

    def backward(self, grad):
        return self.act.backward(grad)

    def get_viz_data(self):
        return {
            "name": self.name,
            "type": "Activation",
            "activation": self.act.name,
        }
