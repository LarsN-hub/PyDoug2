"""
Plot and modify an image array's histogram.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from pydoug_core import util
from pydoug_core.analyze import plot_util
from pydoug_core.registry import register_operation

__all__ = ["plot_histogram"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


@register_operation(
    category="histogram",
    display_name="Plot Histogram",
    mask_params=("Mask",),
    output_kind="value",
    display_result=lambda fig: fig.show(),
)
def plot_histogram(
    imarray: np.ndarray,
    Bin_Count: int = 256,
    Normalize: bool = False,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> Figure:
    """
    Plot an image array's histogram. Convenience wrapper around scikit-image's
    exposure.histogram:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram

    Parameters
    ----------
    imarray: np.ndarray
        The image array to plot the histogram for.
    Bin_Count: int
        The number of bins to use in the histogram. Ignored for integer arrays.
    Normalize: bool
        Whether or not to normalize the counts by the total sum. Default is False.
    Apply_Mask: bool
        Whether or not to apply a mask. Default is False.
    Mask: np.ndarray | None
        (Optional) If Apply Mask is True, only values within the provided mask array
        will be included in the histogram.

    Returns
    -------
    hist_fig: plt.Figure
        The figure containing the histogram plot.
    """

    if Apply_Mask and Mask is not None:
        values: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        values = np.copy(imarray)
    fig, ax = plt.subplots(layout="constrained")
    ax = plot_util.hist_ax(values, ax, Bin_Count, normalize=Normalize)
    ax.set_xlabel("Gray Value", fontsize=15)
    if Normalize:
        ax.set_ylabel("Probability", fontsize=15)
    else:
        ax.set_ylabel("Counts", fontsize=15)
    ax.tick_params("both", labelsize=13)
    return fig
