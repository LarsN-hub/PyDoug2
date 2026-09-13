"""
Utilities for generating plots.
"""

from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from skimage.exposure import histogram

__all__ = ["hist_ax"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


def hist_ax(
    values: np.ndarray,
    hist_ax: Axes,
    nbins: int = 256,
    source_range: Literal["values", "dtype"] = "dtype",
    normalize: bool = False,
) -> Axes:
    """
    Generate an axis with a minimal histogram plot for the given values. Convenience
    wrapper around scikit-image's exposure.histogram function:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram

    Parameters
    ----------
    values: np.ndarray
        The values to generate a histogram for.
    hist_ax: plt.Axes
        The axis to plot the histogram on.
    nbins: int
        The number of bins to use. Default is 256.
    source_range: Literal["values", "dtype"]
        Whether to use the values or their dtype as the source for the x-axis range.
    normalize: bool
        Whether or not to normalize the histogram. Default is False.

    Returns
    -------
    hist_ax: plt.Axes
        The input axis with the histogram plot generated.
    """

    input_range: str = source_range
    if source_range == "values":
        input_range = "image"
    hist, bin_centers = histogram(values, nbins, input_range, normalize)
    hist_ax.hist(bin_centers, bin_centers, weights=hist, edgecolor="k")
    return hist_ax
