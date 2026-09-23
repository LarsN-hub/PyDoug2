"""
Utilities for segmenting images.
"""

from typing import Annotated, Any, Literal

import numpy as np
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_multiotsu,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

__all__ = ["calc_thresholds"]


def calc_thresholds(
    imarray: np.ndarray,
    Method: Literal[
        "Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"
    ] = "Otsu",
    Otsu_Classes: Annotated[int, {"min": 2}] = 2,
    Bin_Count: Annotated[int, {"min": 1}] = 256,
    Max_Iteration_Count: Annotated[int, {"min": 1, "max": 100000}] = 10000,
) -> Any:
    """
    Calculate segmentation thresholds for an image using histogram-based algorithms.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Method: Literal["Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"]
        The algorithm to use for identifying the threshold value(s). Default is Otsu.
    Otsu_Classes: int
        The number of classes to segmented if "Method" is "Otsu".
        Ignored for other methods.
    Bin_Count: int
        The number of bins to use for calculating the histogram.
        Ignored for integer dtypes.
    Max_Iteration_Count: int
        The maximum number of iterations to use for smoothing the histogram.
        Only applied if "Method" is "Minimum".

    Returns
    -------
    thresh_vals: float | int | np.ndarray
        The threshold value(s).
    """

    thresh_vals: Any = None
    if Method == "Isodata":
        thresh_vals = threshold_isodata(imarray, Bin_Count)
    if Method == "Li":
        thresh_vals = threshold_li(imarray)
    if Method == "Mean":
        thresh_vals = threshold_mean(imarray)
    if Method == "Minimum":
        thresh_vals = threshold_minimum(imarray, Bin_Count, Max_Iteration_Count)
    if Method == "Otsu":
        if Otsu_Classes > 2:
            thresh_vals = threshold_multiotsu(imarray, Otsu_Classes, Bin_Count)
        else:
            thresh_vals = threshold_otsu(imarray, Bin_Count)
    if Method == "Triangle":
        thresh_vals = threshold_triangle(imarray, Bin_Count)
    if Method == "Yen":
        thresh_vals = threshold_yen(imarray, Bin_Count)

    return thresh_vals
