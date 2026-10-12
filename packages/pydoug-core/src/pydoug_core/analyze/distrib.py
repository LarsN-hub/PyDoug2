"""
Generate distributions from images.
"""

from typing import Literal

import numpy as np

__all__ = ["hist_distrib"]


def hist_distrib(
    values: np.ndarray,
    nbins: int = 256,
    source_range: Literal["values", "dtype"] = "dtype",
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """
    Generate an histogram for the given values. Convenience wrapper around
    scikit-image's exposure.histogram:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram

    Parameters
    ----------
    values: np.ndarray
        The values to generate a histogram for.
    nbins: int
        The number of bins to use. Default is 256.
    source_range: Literal["values", "dtype"]
        Whether to use the values or their dtype as the source for the x-axis range.
    normalize: bool
        Whether or not to normalize the histogram. Default is False.

    Returns
    -------
    hist_vals: np.ndarray
        The counts corresponding to each bin center in "bin_centers".
    bin_centers: np.ndarray
        The bin centers corresponding to each histogram value in "hist_vals".
    """

    from skimage.exposure import histogram

    from pydoug_core.analyze.measure import stats

    input_range: str = source_range
    if source_range == "values":
        input_range = "image"
    hist_vals, bin_centers = histogram(values, nbins, input_range, normalize)
    stats_dict: dict[str, float | int] = stats(values)
    return hist_vals, bin_centers, stats_dict
