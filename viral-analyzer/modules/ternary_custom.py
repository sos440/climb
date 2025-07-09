"""
Ternary scatter plot with customizations.

This module provides a function to create ternary scatter plots with various customization options, including grid lines, axes labels, and color mapping.
The plot is based on barycentric coordinates, which are converted to Cartesian coordinates for visualization.

I created this module because the existing ternary plot libraries did not meet my specific requirements for customization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from numpy.typing import ArrayLike


vertices = np.array([[0.5, np.sqrt(3) / 2], [0, 0], [1, 0]])


def b2c(abc: ArrayLike) -> np.ndarray:
    """
    Barycentric to Cartesian.

    Supports batch processing.
    """
    abc = np.asarray(abc)
    if abc.ndim == 0:
        return abc
    abc = abc / np.sum(abc, axis=-1, keepdims=True)  # Normalize to ensure they sum to 1
    return np.dot(abc, vertices)


def draw_ternary_plot(
    data,
    values,
    cmap="coolwarm",
    figsize=(10, 8),
    grid_steps=10,
    tick_length=0.02,
    vmin: float | None = None,
    vmax: float | None = None,
    axes_labels=("A", "B", "C"),
    marker_opts: dict | None = None,
):
    """
    Ternary scatter plot

    Parameters:
    -----------
    data : array-like of shape (n, 3)
    values : array-like of shape (n,), for coloring the points
    cmap : str or Colormap, colormap to use for the points
    figsize : tuple, size of the figure
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # 1. Draw background grid
    for i in range(1, grid_steps):
        r = i / grid_steps
        # Parallel lines for B-C
        ax.plot(*b2c([[r, 1 - r, 0], [r, 0, 1 - r]]).T, "lightgray", linewidth=0.5, alpha=0.5)
        # Parallel lines for A-B
        ax.plot(*b2c([[0, r, 1 - r], [1 - r, r, 0]]).T, "lightgray", linewidth=0.5, alpha=0.5)
        # Parallel lines for A-C
        ax.plot(*b2c([[1 - r, 0, r], [0, 1 - r, r]]).T, "lightgray", linewidth=0.5, alpha=0.5)

    # 2. Draw triangle outline
    triangle = Polygon(vertices, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)

    # 3. Axes labels
    ax.text(*b2c(np.array([0.5, 0.55, -0.05])), axes_labels[0], ha="right", va="center", fontsize=14, fontweight="bold")
    ax.text(
        *b2c(np.array([-0.05, 0.55, 0.5])), axes_labels[1], ha="center", va="center", fontsize=14, fontweight="bold"
    )
    ax.text(*b2c(np.array([0.5, -0.05, 0.55])), axes_labels[2], ha="left", va="center", fontsize=14, fontweight="bold")

    # 4. Draw ticks on the edges
    tl = tick_length
    for i in range(grid_steps + 1):
        r = i / grid_steps
        ax.plot(*b2c([[r, 1 - r, 0], [r, 1 - r + tl, -tl]]).T, "k-", linewidth=1)
        ax.plot(*b2c([[0, r, 1 - r], [-tl, r, 1 - r + tl]]).T, "k-", linewidth=1)
        ax.plot(*b2c([[1 - r, 0, r], [1 - r + tl, -tl, r]]).T, "k-", linewidth=1)

    # 5. Draw the data points
    data = np.asarray(data)
    points = b2c(data)
    x_coords, y_coords = points[:, 0], points[:, 1]

    # Scatter plot
    marker_opts = marker_opts or {}
    marker_opts = dict(s=100, edgecolors="black", linewidth=1, alpha=0.8) | marker_opts
    scatter = ax.scatter(x_coords, y_coords, c=values, cmap=cmap, **marker_opts)

    # 6. Add colorbar
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mapper, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
    cbar.set_label("Value", rotation=0, labelpad=5)

    # Axes limits and aspect
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.axis("off")

    plt.tight_layout()
    return fig, ax
