"""
Utilities for generating plots.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

__all__ = ["hist_ax"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


def hist_ax(
    hist_vals: np.ndarray,
    bin_centers: np.ndarray,
    hist_ax: Axes,
    x_label: str = "Gray Value",
    y_label: str = "Counts",
    y_max: float = 0,
    x_lims: tuple[float, float] | None = None,
) -> Axes:
    """
    Generate an axis with a minimal histogram plot for the given values. Convenience
    wrapper around scikit-image's exposure.histogram function:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram

    Parameters
    ----------
    hist_vals: np.ndarray
        The values to generate a histogram for.
    bin_centers: np.ndarray
        The bin centers for each value in "hist_vals".
    hist_ax: Axes
        The axis to add the histogram to.
    y_max: float
        The maximum y-axis value to plot. Ignored if 0.
    x_lims: tuple[float, float] | None
        The x-axis limits (optional).

    Returns
    -------
    hist_ax: plt.Axes
        The input axis with the histogram plot generated.
    """

    hist_ax.hist(bin_centers, bin_centers, weights=hist_vals, edgecolor="k")  # type: ignore
    x_lims = (min(x_lims), max(x_lims)) if x_lims else None
    hist_ax.set_xlim(x_lims)
    if y_max:
        hist_ax.set_ylim(0, y_max)
    hist_ax.set_xlabel(x_label, fontsize=15)
    hist_ax.set_ylabel(y_label, fontsize=15)
    hist_ax.tick_params("both", labelsize=13)
    return hist_ax
