"""
Visualizer: unified dashboard for neural network visualization.

Usage:
    viz = Visualizer(model)
    viz.plot_all(X, y)           # Save all plots to disk
    viz.show_training()           # Just training curves
    viz.show_weights()            # Weight/gradient heatmaps
    viz.show_architecture()       # Network graph
    viz.show_activations(X[:5])   # Activation distributions
    viz.show_decision_boundary(X, y)  # 2D decision boundary
    viz.show_gradient_flow()      # Gradient flow bar chart
    viz.show_loss_landscape(X, y) # Loss landscape
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plots import (
    plot_training_curves,
    plot_weight_heatmap,
    plot_gradient_flow,
    plot_activation_distributions,
    plot_network_graph,
    plot_loss_landscape,
    plot_decision_boundary,
    plot_layer_activations,
)

DARK_BG = "#0d1117"
TEXT_COLOR = "#c9d1d9"


class Visualizer:
    """
    One-stop visualization dashboard for a NeuralNetwork instance.

    All plot methods return the matplotlib Figure so you can call
    fig.savefig(...) or plt.show() yourself, or use save_dir to
    auto-save everything.
    """

    def __init__(self, model, save_dir="viz_output"):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # Individual plots
    # ─────────────────────────────────────────────────────────────

    def show_training(self, save=True, smooth=5):
        fig = plot_training_curves(self.model, smooth=smooth)
        if save:
            self._save(fig, "training_curves.png")
        return fig

    def show_weights(self, save=True):
        fig = plot_weight_heatmap(self.model)
        if save:
            self._save(fig, "weight_heatmap.png")
        return fig

    def show_gradient_flow(self, save=True):
        fig = plot_gradient_flow(self.model)
        if save:
            self._save(fig, "gradient_flow.png")
        return fig

    def show_architecture(self, save=True, max_neurons=12):
        fig = plot_network_graph(self.model, max_neurons=max_neurons)
        if save:
            self._save(fig, "architecture.png")
        return fig

    def show_activations(self, X_sample, save=True):
        fig = plot_activation_distributions(self.model, X_sample)
        if save:
            self._save(fig, "activation_distributions.png")
        return fig

    def show_layer_activations(self, x_single, save=True):
        fig = plot_layer_activations(self.model, x_single)
        if save:
            self._save(fig, "layer_activations.png")
        return fig

    def show_decision_boundary(self, X, y, save=True, resolution=200):
        assert X.shape[1] == 2, "Decision boundary requires 2D input features."
        fig = plot_decision_boundary(self.model, X, y, resolution=resolution)
        if save:
            self._save(fig, "decision_boundary.png")
        return fig

    def show_loss_landscape(self, X, y, layer_idx=0, save=True,
                             resolution=30, perturbation=0.5):
        fig = plot_loss_landscape(
            self.model, X, y,
            layer_idx=layer_idx,
            resolution=resolution,
            perturbation=perturbation,
        )
        if save:
            self._save(fig, "loss_landscape.png")
        return fig

    def show_activation_curves(self, save=True):
        """Plot the mathematical curve of every activation function in the network."""
        from ..core.activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, Linear
        from .plots import DARK_BG, PANEL_BG, TEXT_COLOR, GRID_COLOR, PALETTE, _dark_fig, _style_ax

        acts = []
        for layer in self.model.layers:
            if hasattr(layer, "activation"):
                acts.append(layer.activation)
            elif hasattr(layer, "act"):
                acts.append(layer.act)

        unique_acts = list({type(a).__name__: a for a in acts}.values())
        if not unique_acts:
            unique_acts = [ReLU(), Sigmoid(), Tanh(), Linear()]

        n = len(unique_acts)
        fig = _dark_fig(figsize=(n * 4, 4))
        for i, act in enumerate(unique_acts):
            ax = fig.add_subplot(1, n, i + 1)
            try:
                x, y = act.get_curve_points()
                xd, yd = act.get_derivative_points()
                ax.plot(x, y, color=PALETTE[i % len(PALETTE)], lw=2.5, label="f(x)")
                ax.plot(xd, yd, color=PALETTE[(i + 2) % len(PALETTE)],
                        lw=1.5, linestyle="--", label="f'(x)", alpha=0.8)
                ax.axhline(0, color=GRID_COLOR, lw=0.8)
                ax.axvline(0, color=GRID_COLOR, lw=0.8)
            except Exception:
                pass
            _style_ax(ax, act.name, "x", "y")
            ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8)
            ax.set_ylim(-1.5, 1.5)

        fig.suptitle("Activation Functions & Derivatives", color=TEXT_COLOR, fontsize=13)
        fig.tight_layout()
        if save:
            self._save(fig, "activation_curves.png")
        return fig

    def show_weight_evolution(self, layer_name=None, save=True):
        """
        Plot how weights evolved during training (using stored history).
        """
        from .plots import DARK_BG, PANEL_BG, TEXT_COLOR, PALETTE, _dark_fig, _style_ax
        from matplotlib.gridspec import GridSpec

        dense_layers = [l for l in self.model.layers if hasattr(l, "W") and l.W is not None]
        if layer_name:
            dense_layers = [l for l in dense_layers if l.name == layer_name]
        if not dense_layers:
            fig, ax = plt.subplots(facecolor=DARK_BG)
            ax.text(0.5, 0.5, "No weight history found.", ha="center", va="center", color=TEXT_COLOR)
            return fig

        n = len(dense_layers)
        fig = _dark_fig(figsize=(n * 6, 8))
        gs = GridSpec(2, n, figure=fig, hspace=0.4)

        for i, layer in enumerate(dense_layers):
            hist = layer.W._history
            ghist = layer.W._grad_history

            ax1 = fig.add_subplot(gs[0, i])
            if hist:
                steps = list(range(len(hist)))
                means = [h["mean"] for h in hist]
                stds = [h["std"] for h in hist]
                norms = [h["norm"] for h in hist]
                ax1.plot(steps, means, color=PALETTE[0], lw=2, label="Mean W")
                ax1.fill_between(steps,
                                  [m - s for m, s in zip(means, stds)],
                                  [m + s for m, s in zip(means, stds)],
                                  alpha=0.2, color=PALETTE[0])
                ax1.plot(steps, norms, color=PALETTE[1], lw=1.5, linestyle="--", label="‖W‖")
            _style_ax(ax1, f"{layer.name} — Weight Stats", "Update", "Value")
            ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8)

            ax2 = fig.add_subplot(gs[1, i])
            if ghist:
                gsteps = list(range(len(ghist)))
                gnorms = [h["norm"] for h in ghist]
                gmeans = [h["mean"] for h in ghist]
                ax2.plot(gsteps, gnorms, color=PALETTE[2], lw=2, label="‖∇W‖")
                ax2.plot(gsteps, gmeans, color=PALETTE[3], lw=1.5, linestyle="--", label="Mean ∇W")
                ax2.set_yscale("log")
            _style_ax(ax2, "Gradient Stats", "Update", "Value")
            ax2.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8)

        fig.suptitle("Weight Evolution During Training", color=TEXT_COLOR, fontsize=13)
        fig.tight_layout()
        if save:
            self._save(fig, "weight_evolution.png")
        return fig

    # ─────────────────────────────────────────────────────────────
    # Combined reports
    # ─────────────────────────────────────────────────────────────

    def plot_all(self, X, y, X_sample_size=256):
        """
        Generate and save every available plot. Returns dict of {name: fig}.
        """
        n_sample = min(X_sample_size, X.shape[0])
        idx = np.random.choice(X.shape[0], n_sample, replace=False)
        X_s = X[idx]

        figs = {}
        print("📊 Generating visualizations...")

        figs["training"] = self.show_training()
        print("  ✓ Training curves")

        figs["architecture"] = self.show_architecture()
        print("  ✓ Architecture graph")

        figs["weights"] = self.show_weights()
        print("  ✓ Weight heatmaps")

        figs["gradient_flow"] = self.show_gradient_flow()
        print("  ✓ Gradient flow")

        figs["activations"] = self.show_activations(X_s)
        print("  ✓ Activation distributions")

        figs["activation_curves"] = self.show_activation_curves()
        print("  ✓ Activation function curves")

        figs["weight_evolution"] = self.show_weight_evolution()
        print("  ✓ Weight evolution")

        figs["layer_activations"] = self.show_layer_activations(X[0])
        print("  ✓ Per-neuron activations")

        if X.shape[1] == 2:
            figs["decision_boundary"] = self.show_decision_boundary(X, y)
            print("  ✓ Decision boundary")

        try:
            figs["loss_landscape"] = self.show_loss_landscape(X_s, y[idx])
            print("  ✓ Loss landscape")
        except Exception as e:
            print(f"  ⚠  Loss landscape skipped: {e}")

        print(f"\n✅ All plots saved to '{self.save_dir}/'")
        return figs

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _save(self, fig, filename):
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=DARK_BG, edgecolor="none")
        plt.close(fig)
        return path

    def print_summary(self):
        print(self.model.get_architecture_summary())
