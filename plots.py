"""
Individual visualization functions.
Each function takes a NeuralNetwork (or its viz state) and returns a matplotlib Figure.
All functions use a consistent dark theme.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT1 = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#7ee787"
ACCENT4 = "#d2a8ff"
ACCENT5 = "#ffa657"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#30363d"

PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
           "#39d353", "#ff7b72", "#79c0ff", "#e3b341", "#bc8cff"]


def _dark_fig(figsize=(12, 6)):
    fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
    return fig


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR)


# ──────────────────────────────────────────────────────────────────────────────
# Training Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(model_or_history, smooth=5):
    """
    Plot train/val loss and accuracy over epochs.
    model_or_history: NeuralNetwork or its .history dict
    """
    if hasattr(model_or_history, "history"):
        h = model_or_history.history
    else:
        h = model_or_history

    train_loss = h.get("train_loss", [])
    val_loss = h.get("val_loss", [])
    train_acc = h.get("train_acc", [])
    val_acc = h.get("val_acc", [])
    batch_losses = h.get("batch_losses", [])

    fig = _dark_fig(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # ── Panel 1: per-epoch loss ──
    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1, "Loss per Epoch", "Epoch", "Loss")
    epochs = list(range(1, len(train_loss) + 1))
    if train_loss:
        ax1.plot(epochs, train_loss, color=ACCENT1, lw=2, label="Train")
    if val_loss:
        ax1.plot(range(1, len(val_loss)+1), val_loss, color=ACCENT2,
                 lw=2, linestyle="--", label="Val")
    if train_loss or val_loss:
        ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)

    # ── Panel 2: accuracy ──
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, "Accuracy per Epoch", "Epoch", "Accuracy")
    if train_acc:
        ax2.plot(range(1, len(train_acc)+1), train_acc, color=ACCENT3, lw=2, label="Train")
    if val_acc:
        ax2.plot(range(1, len(val_acc)+1), val_acc, color=ACCENT5,
                 lw=2, linestyle="--", label="Val")
    ax2.set_ylim([0, 1.05])
    if train_acc or val_acc:
        ax2.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)

    # ── Panel 3: batch loss with smoothing ──
    ax3 = fig.add_subplot(gs[2])
    _style_ax(ax3, "Batch Loss (raw + smoothed)", "Batch", "Loss")
    if batch_losses:
        x = list(range(len(batch_losses)))
        ax3.plot(x, batch_losses, color=ACCENT1, alpha=0.3, lw=0.8, label="Raw")
        if len(batch_losses) >= smooth:
            kernel = np.ones(smooth) / smooth
            smoothed = np.convolve(batch_losses, kernel, mode="valid")
            ax3.plot(range(smooth - 1, len(batch_losses)), smoothed,
                     color=ACCENT1, lw=2, label=f"Smooth({smooth})")
        ax3.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)

    fig.suptitle("Training History", color=TEXT_COLOR, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Weight Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_weight_heatmap(model):
    """
    Display weight matrices and gradient matrices for all Dense layers.
    """
    dense_layers = [l for l in model.layers if hasattr(l, "W") and l.W is not None]
    if not dense_layers:
        fig, ax = plt.subplots(facecolor=DARK_BG)
        ax.text(0.5, 0.5, "No Dense layers with weights",
                ha="center", va="center", color=TEXT_COLOR)
        return fig

    n = len(dense_layers)
    fig = _dark_fig(figsize=(max(14, n * 5), 8))
    gs = GridSpec(2, n, figure=fig, hspace=0.4, wspace=0.3)

    for i, layer in enumerate(dense_layers):
        W = layer.W.data
        G = layer.W.grad

        # Weight heatmap
        ax_w = fig.add_subplot(gs[0, i])
        vmax = max(abs(W.min()), abs(W.max())) + 1e-8
        im_w = ax_w.imshow(W, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
        _style_ax(ax_w, f"{layer.name}\nWeights {W.shape}")
        plt.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)

        # Gradient heatmap
        ax_g = fig.add_subplot(gs[1, i])
        vmax_g = max(abs(G.min()), abs(G.max())) + 1e-8
        im_g = ax_g.imshow(G, aspect="auto", cmap="PuOr",
                           vmin=-vmax_g, vmax=vmax_g, interpolation="nearest")
        _style_ax(ax_g, f"Gradients ∂L/∂W")
        plt.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)

    fig.suptitle("Weight & Gradient Heatmaps", color=TEXT_COLOR, fontsize=14)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Flow
# ──────────────────────────────────────────────────────────────────────────────

def plot_gradient_flow(model):
    """
    Bar chart of mean absolute gradient per layer — detects vanishing/exploding gradients.
    """
    names, means, maxs = [], [], []
    for layer in model.layers:
        for param in layer.get_params():
            if param.grad is not None and np.any(param.grad != 0):
                names.append(param.name)
                means.append(float(np.mean(np.abs(param.grad))))
                maxs.append(float(np.max(np.abs(param.grad))))

    fig = _dark_fig(figsize=(max(10, len(names) * 1.2), 6))
    ax = fig.add_subplot(111)
    _style_ax(ax, "Gradient Flow (Mean |∇|) per Parameter", "Parameter", "|Gradient|")

    if names:
        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w/2, means, w, label="Mean |∇|", color=ACCENT1, alpha=0.85)
        bars2 = ax.bar(x + w/2, maxs, w, label="Max |∇|", color=ACCENT2, alpha=0.85)

        # Highlight near-zero gradients (vanishing)
        for bar, v in zip(bars1, means):
            if v < 1e-6:
                bar.set_edgecolor("#ff4040")
                bar.set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", color=TEXT_COLOR, fontsize=8)
        ax.set_yscale("log")
        ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
        ax.axhline(y=1e-5, color=ACCENT2, linestyle=":", alpha=0.7, label="Vanishing threshold")
        ax.axhline(y=1.0, color=ACCENT5, linestyle=":", alpha=0.7, label="Exploding threshold")
    else:
        ax.text(0.5, 0.5, "No gradients recorded yet.\nRun training first.",
                ha="center", va="center", color=TEXT_COLOR, transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Activation Distributions
# ──────────────────────────────────────────────────────────────────────────────

def plot_activation_distributions(model, X_sample):
    """
    Histogram of activations at each layer for a sample input.
    """
    # Collect activations via a forward pass
    activations = {}
    out = X_sample
    for layer in model.layers:
        out = layer.forward(out, training=False)
        if hasattr(layer, "_output_cache") and layer._output_cache is not None:
            activations[layer.name] = out.copy()

    n = len(activations)
    if n == 0:
        fig, ax = plt.subplots(facecolor=DARK_BG)
        ax.text(0.5, 0.5, "No layers to visualize", ha="center", va="center", color=TEXT_COLOR)
        return fig

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig = _dark_fig(figsize=(cols * 4, rows * 3.5))

    for idx, (name, act) in enumerate(activations.items()):
        ax = fig.add_subplot(rows, cols, idx + 1)
        vals = act.ravel()
        color = PALETTE[idx % len(PALETTE)]
        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="none")
        _style_ax(ax, name, "Value", "Count")

        dead = np.mean(vals == 0)
        stats = f"μ={np.mean(vals):.3f}\nσ={np.std(vals):.3f}\ndead={dead:.1%}"
        ax.text(0.97, 0.97, stats, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                color=TEXT_COLOR, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, alpha=0.7))

    fig.suptitle("Activation Distributions", color=TEXT_COLOR, fontsize=14)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Network Graph
# ──────────────────────────────────────────────────────────────────────────────

def plot_network_graph(model, max_neurons=12):
    """
    Schematic diagram of the network topology with animated-style neuron circles.
    """
    fig = _dark_fig(figsize=(max(12, len(model.layers) * 2), 8))
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Determine layer sizes
    layer_sizes = []
    layer_labels = []
    for layer in model.layers:
        if hasattr(layer, "units"):
            layer_sizes.append(min(layer.units, max_neurons))
            layer_labels.append(layer.name)
        elif hasattr(layer, "_output_cache") and layer._output_cache is not None:
            s = min(layer._output_cache.shape[-1], max_neurons)
            layer_sizes.append(s)
            layer_labels.append(layer.name)
        else:
            layer_sizes.append(1)
            layer_labels.append(layer.name)

    if not layer_sizes:
        ax.text(0.5, 0.5, "Network not built yet.\nRun a forward pass first.",
                ha="center", va="center", color=TEXT_COLOR, transform=ax.transAxes)
        return fig

    # Layout
    n_layers = len(layer_sizes)
    x_positions = np.linspace(0.1, 0.9, n_layers)
    max_size = max(layer_sizes) if layer_sizes else 1

    neuron_positions = []
    for li, (lx, lsize) in enumerate(zip(x_positions, layer_sizes)):
        ys = np.linspace(0.1, 0.9, lsize) if lsize > 1 else [0.5]
        neuron_positions.append(list(zip([lx] * lsize, ys)))

    # Draw connections (thin, faded)
    if len(neuron_positions) > 1:
        for li in range(len(neuron_positions) - 1):
            for x1, y1 in neuron_positions[li]:
                for x2, y2 in neuron_positions[li + 1]:
                    ax.plot([x1, x2], [y1, y2], color=GRID_COLOR,
                            alpha=0.25, lw=0.5, zorder=1)

    # Draw neurons
    radius = 0.018
    for li, (lx, positions) in enumerate(zip(x_positions, neuron_positions)):
        color = PALETTE[li % len(PALETTE)]
        for (nx, ny) in positions:
            circle = plt.Circle((nx, ny), radius, color=color,
                                  alpha=0.9, zorder=3)
            ax.add_patch(circle)
            circle_outline = plt.Circle((nx, ny), radius, color=TEXT_COLOR,
                                         fill=False, lw=0.5, alpha=0.4, zorder=4)
            ax.add_patch(circle_outline)

        # Ellipsis if truncated
        orig_size = layer_sizes[li]
        actual = model.layers[li]
        actual_size = getattr(actual, "units", orig_size)
        if actual_size > max_neurons:
            ax.text(lx, 0.05, f"... ({actual_size})",
                    ha="center", va="center", color=TEXT_COLOR, fontsize=7)

        # Layer label
        ax.text(lx, 0.02, layer_labels[li], ha="center", va="top",
                color=TEXT_COLOR, fontsize=8, rotation=30,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_BG, alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Network Architecture", color=TEXT_COLOR, fontsize=14, pad=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Loss Landscape
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_landscape(model, X, y, layer_idx=0, param="W",
                        resolution=30, perturbation=0.5):
    """
    2D loss landscape by perturbing a layer's weights along two random directions.
    """
    layers_with_W = [l for l in model.layers if hasattr(l, "W") and l.W is not None]
    if not layers_with_W or layer_idx >= len(layers_with_W):
        fig, ax = plt.subplots(facecolor=DARK_BG)
        ax.text(0.5, 0.5, "No weights to perturb", ha="center", va="center", color=TEXT_COLOR)
        return fig

    layer = layers_with_W[layer_idx]
    W0 = layer.W.data.copy()

    # Random perturbation directions
    d1 = np.random.randn(*W0.shape)
    d1 /= (np.linalg.norm(d1) + 1e-8)
    d2 = np.random.randn(*W0.shape)
    d2 -= d1 * np.dot(d1.ravel(), d2.ravel())
    d2 /= (np.linalg.norm(d2) + 1e-8)

    alphas = np.linspace(-perturbation, perturbation, resolution)
    betas = np.linspace(-perturbation, perturbation, resolution)
    Z = np.zeros((resolution, resolution))

    sample_size = min(200, X.shape[0])
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    Xs, ys = X[idx], y[idx]

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            layer.W.data = W0 + a * d1 + b * d2
            pred = model._forward(Xs, training=False)
            Z[i, j] = float(model.loss_fn.forward(pred, ys))

    # Restore weights
    layer.W.data = W0.copy()

    fig = _dark_fig(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # Contour
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PANEL_BG)
    A, B = np.meshgrid(alphas, betas)
    contour = ax1.contourf(A, B, Z.T, levels=30, cmap="inferno")
    ax1.contour(A, B, Z.T, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1).ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    ax1.scatter([0], [0], color=ACCENT3, s=80, zorder=5, label="Current pos")
    _style_ax(ax1, f"Loss Landscape ({layer.name})", "Direction 1", "Direction 2")
    ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR)

    # 3D-style surface with matplotlib
    ax2 = fig.add_subplot(gs[1], projection="3d")
    ax2.set_facecolor(DARK_BG)
    surf = ax2.plot_surface(A, B, Z.T, cmap="inferno", alpha=0.85, linewidth=0)
    ax2.set_xlabel("Dir 1", color=TEXT_COLOR)
    ax2.set_ylabel("Dir 2", color=TEXT_COLOR)
    ax2.set_zlabel("Loss", color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax2.set_title("3D Loss Surface", color=TEXT_COLOR)

    fig.suptitle("Loss Landscape", color=TEXT_COLOR, fontsize=14)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Decision Boundary (2D inputs only)
# ──────────────────────────────────────────────────────────────────────────────

def plot_decision_boundary(model, X, y, resolution=200, title="Decision Boundary"):
    """
    Only works for 2D input data. Shows the class decision regions.
    """
    assert X.shape[1] == 2, "Decision boundary only for 2D input"

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)

    if preds.shape[1] > 1:
        Z = np.argmax(preds, axis=1)
        conf = np.max(preds, axis=1)
    else:
        Z = (preds.ravel() > 0.5).astype(int)
        conf = np.abs(preds.ravel() - 0.5) * 2

    Z = Z.reshape(xx.shape)
    conf = conf.reshape(xx.shape)

    n_classes = int(Z.max()) + 1
    colors_bg = [PALETTE[i % len(PALETTE)] for i in range(n_classes)]
    cmap_bg = mcolors.ListedColormap(colors_bg)

    fig = _dark_fig(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)

    # Confidence shading
    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.3)
    ax.contourf(xx, yy, conf, levels=20, cmap="Greys", alpha=0.15)
    ax.contour(xx, yy, Z, levels=n_classes - 1, colors="white", linewidths=1.5, alpha=0.7)

    # Data points
    if y.ndim > 1:
        y_int = np.argmax(y, axis=1)
    else:
        y_int = y.astype(int)

    for cls in range(n_classes):
        mask = y_int == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=PALETTE[cls % len(PALETTE)], s=30, edgecolors="white",
                   linewidths=0.5, alpha=0.9, zorder=5, label=f"Class {cls}")

    _style_ax(ax, title, "Feature 1", "Feature 2")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Layer-by-Layer Activation Maps
# ──────────────────────────────────────────────────────────────────────────────

def plot_layer_activations(model, x_single):
    """
    For a single input, show the activation value at every neuron in every layer.
    """
    if x_single.ndim == 1:
        x_single = x_single.reshape(1, -1)

    activations = {}
    out = x_single
    for layer in model.layers:
        out = layer.forward(out, training=False)
        activations[layer.name] = out.ravel().copy()

    n = len(activations)
    fig = _dark_fig(figsize=(max(12, n * 3), 4))
    gs = GridSpec(1, n, figure=fig, wspace=0.35)

    for i, (name, vals) in enumerate(activations.items()):
        ax = fig.add_subplot(gs[i])
        color = PALETTE[i % len(PALETTE)]
        n_neurons = len(vals)

        if n_neurons <= 64:
            ax.bar(range(n_neurons), vals, color=color, alpha=0.85, edgecolor="none")
            ax.axhline(0, color=TEXT_COLOR, lw=0.5, alpha=0.5)
        else:
            # For larger layers, use a heatmap
            side = int(np.ceil(np.sqrt(n_neurons)))
            padded = np.zeros(side * side)
            padded[:n_neurons] = vals
            ax.imshow(padded.reshape(side, side), cmap="RdBu_r", aspect="auto")

        _style_ax(ax, name, "Neuron", "Value")
        ax.set_title(f"{name}\n({n_neurons} neurons)", color=TEXT_COLOR, fontsize=9)

    fig.suptitle("Per-Neuron Activations (single input)", color=TEXT_COLOR, fontsize=13)
    fig.tight_layout()
    return fig
