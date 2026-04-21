import numpy as np
import json
import time


class NeuralNetwork:
    """
    The central model class.

    Usage:
        model = NeuralNetwork()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=Adam(lr=0.001), loss=CrossEntropy())
        history = model.fit(X_train, y_train, epochs=20, batch_size=32)
    """

    def __init__(self, name="NeuralNetwork"):
        self.name = name
        self.layers = []
        self.optimizer = None
        self.loss_fn = None
        self._built = False
        self._training = False

        # Full training history for visualization
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epoch_times": [],
            "batch_losses": [],
            "step": 0,
            "epoch": 0,
        }
        # Snapshots: full layer state captured periodically
        self._snapshots = []
        self._snapshot_interval = 1  # epochs between snapshots

    # ──────────────────────────────────────────────────────────
    # Building
    # ──────────────────────────────────────────────────────────

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
        return self

    def compile(self, optimizer, loss):
        """Set optimizer and loss function."""
        self.optimizer = optimizer
        self.loss_fn = loss

    # ──────────────────────────────────────────────────────────
    # Forward / Backward
    # ──────────────────────────────────────────────────────────

    def _forward(self, x, training=True):
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def _backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def _get_all_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def _zero_grads(self):
        for p in self._get_all_params():
            p.zero_grad()

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────

    def fit(self, X, y, epochs=10, batch_size=32,
            validation_data=None, verbose=True,
            snapshot_interval=1, callbacks=None):
        """
        Train the network.

        Parameters
        ----------
        X, y : numpy arrays
        epochs : int
        batch_size : int
        validation_data : tuple (X_val, y_val) or None
        verbose : bool
        snapshot_interval : int — save a full state snapshot every N epochs
        callbacks : list of callables f(epoch, model) called after each epoch
        """
        self._snapshot_interval = snapshot_interval
        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            t_start = time.time()
            self.history["epoch"] = epoch

            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuf[i:i + batch_size]
                y_batch = y_shuf[i:i + batch_size]

                self._zero_grads()
                pred = self._forward(X_batch, training=True)
                loss_val = self.loss_fn.forward(pred, y_batch)
                grad = self.loss_fn.backward()
                self._backward(grad)
                self.optimizer.step(self._get_all_params())

                epoch_losses.append(float(loss_val))
                self.history["batch_losses"].append(float(loss_val))
                self.history["step"] += 1

            # Keep batch_losses from bloating
            if len(self.history["batch_losses"]) > 5000:
                self.history["batch_losses"] = self.history["batch_losses"][-5000:]

            mean_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(mean_loss)

            # Training accuracy (if classification)
            train_acc = self._compute_accuracy(X, y)
            self.history["train_acc"].append(train_acc)

            # Validation
            val_loss, val_acc = None, None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self._forward(X_val, training=False)
                val_loss = float(self.loss_fn.forward(val_pred, y_val))
                val_acc = self._compute_accuracy(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t_start
            self.history["epoch_times"].append(elapsed)

            # Snapshot
            if epoch % snapshot_interval == 0:
                self._take_snapshot(epoch)

            # Callbacks
            if callbacks:
                for cb in callbacks:
                    cb(epoch, self)

            if verbose:
                msg = (f"Epoch {epoch:>3}/{epochs}  "
                       f"loss={mean_loss:.4f}  acc={train_acc:.3f}")
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                msg += f"  ({elapsed:.2f}s)"
                print(msg)

        return self.history

    def predict(self, X, batch_size=256):
        outputs = []
        for i in range(0, X.shape[0], batch_size):
            out = self._forward(X[i:i + batch_size], training=False)
            outputs.append(out)
        return np.vstack(outputs)

    def _compute_accuracy(self, X, y):
        try:
            pred = self.predict(X)
            if pred.shape[-1] > 1:
                pred_labels = np.argmax(pred, axis=1)
                if y.ndim > 1:
                    true_labels = np.argmax(y, axis=1)
                else:
                    true_labels = y.astype(int)
            else:
                pred_labels = (pred.ravel() > 0.5).astype(int)
                true_labels = y.ravel().astype(int)
            return float(np.mean(pred_labels == true_labels))
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────
    # Visualization data extraction
    # ──────────────────────────────────────────────────────────

    def _take_snapshot(self, epoch):
        snapshot = {
            "epoch": epoch,
            "train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "layers": [layer.get_viz_data() for layer in self.layers],
            "optimizer": self.optimizer.get_viz_data() if self.optimizer else {},
        }
        self._snapshots.append(snapshot)
        if len(self._snapshots) > 200:
            self._snapshots = self._snapshots[-200:]

    def get_full_viz_state(self):
        """Return the complete state dict for the visualizer."""
        return {
            "name": self.name,
            "history": {
                "train_loss": self.history["train_loss"],
                "val_loss": self.history["val_loss"],
                "train_acc": self.history["train_acc"],
                "val_acc": self.history["val_acc"],
                "batch_losses": self.history["batch_losses"][-500:],
                "epoch_times": self.history["epoch_times"],
                "epoch": self.history["epoch"],
                "step": self.history["step"],
            },
            "layers": [layer.get_viz_data() for layer in self.layers],
            "optimizer": self.optimizer.get_viz_data() if self.optimizer else {},
            "loss_fn": {
                "name": self.loss_fn.name,
                "history": self.loss_fn.get_history()[-500:],
            } if self.loss_fn else {},
            "snapshots": self._snapshots,
        }

    def get_architecture_summary(self):
        """Text summary of the network architecture."""
        lines = [f"{'=' * 60}", f"  {self.name}", f"{'=' * 60}"]
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = [p for p in layer.get_params() if p is not None]
            n_params = sum(p.data.size for p in params if p.data is not None)
            total_params += n_params
            lines.append(f"  [{i}] {layer.name:<25}  params={n_params:>8,}")
        lines.append(f"{'─' * 60}")
        lines.append(f"  Total parameters: {total_params:,}")
        lines.append(f"  Optimizer: {self.optimizer.name if self.optimizer else 'None'}")
        lines.append(f"  Loss: {self.loss_fn.name if self.loss_fn else 'None'}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def save_state(self, path):
        """Save weights to a JSON file."""
        state = {}
        for i, layer in enumerate(self.layers):
            for param in layer.get_params():
                key = f"layer{i}/{param.name}"
                state[key] = param.data.tolist()
        with open(path, "w") as f:
            json.dump(state, f)
        print(f"Saved to {path}")

    def load_state(self, path):
        """Load weights from a JSON file."""
        with open(path) as f:
            state = json.load(f)
        for i, layer in enumerate(self.layers):
            for param in layer.get_params():
                key = f"layer{i}/{param.name}"
                if key in state:
                    param.data = np.array(state[key])
        print(f"Loaded from {path}")
