"""
Intensity-based thresholding operations to segment images.
"""

from typing import Annotated, Any, Literal

import numpy as np
from skimage.filters import apply_hysteresis_threshold

from pydoug_core import util
from pydoug_core.registry import register_operation
from pydoug_core.segment import segment_util

__all__ = ["threshold_array", "hist_threshold", "hysteresis"]


def threshold_array(
    imarray: np.ndarray,
    Range: tuple[float | int, float | int],
) -> np.ndarray:
    """
    Segment an image by assigning pixels within the specified thresholds to be white
    while others are assigned black.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Range: tuple[int, int]
        The range of intensities to include in the white phase.

    Returns
    -------
    thresh_array: np.ndarray
        The thresholded image array.
    """

    thresh_array: np.ndarray = np.zeros(imarray.shape, np.uint8)
    thresh_array[(imarray >= min(Range)) & (imarray <= max(Range))] = 255
    return thresh_array


@register_operation(
    category="segmentation",
    display_name="Interactive Threshold",
    live_preview=True,
    widget_options={"Range": {"widget_type": "RangeSlider", "min": 0, "max": 255}},
)
def interactive_threshold(
    imarray: np.ndarray,
    Range: tuple[int, int] = (0, 255),
) -> np.ndarray:
    """
    Segment an image by assigning pixels within the specified thresholds to be white
    while others are assigned black. Check the "Live Preview" box to view a live
    preview of the segmentation (only available for uint8 dtype).

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Range: tuple[int, int]
        The range of intensities to include in the white phase.

    Returns
    -------
    thresh_array: np.ndarray
        The thresholded image array.
    """

    return threshold_array(imarray, Range)


@register_operation(
    category="segmentation",
    display_name="Manual Threshold",
)
def manual_threshold(
    imarray: np.ndarray,
    Range: tuple[
        Annotated[float, {"max": 65535, "min": -32768, "step": 0.0001}],
        Annotated[float, {"max": 65535, "min": -32768, "step": 0.0001}],
    ] = (0, 255),
) -> np.ndarray:
    """
    Segment an image by assigning pixels within the specified thresholds to be white
    while others are assigned black. Handles all dtypes unlike Interactive Threshold.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Range: tuple[int, int]
        The range of intensities to include in the white phase.

    Returns
    -------
    thresh_array: np.ndarray
        The thresholded image array.
    """

    return threshold_array(imarray, Range)


@register_operation(
    category="segmentation",
    display_name="Histogram Threshold",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def hist_threshold(
    imarray: np.ndarray,
    Method: Literal[
        "Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"
    ] = "Otsu",
    Otsu_Classes: Annotated[int, {"min": 2}] = 2,
    Bin_Count: Annotated[int, {"min": 1}] = 256,
    Max_Iteration_Count: Annotated[int, {"min": 1, "max": 100000}] = 10000,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Segment an image using histogram-based algorithms. Convenience wrapper around the
    histogram threshold functions in scikit-image's filters package:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_li
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_mean
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_minimum
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_triangle
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_yen

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
    Apply_Mask: bool
        Whether or not to apply a mask. Default is False.
    Mask: np.ndarray | None
        (Optional) If Apply Mask is True, only values within the provided mask array
        will be included in the histogram.

    Returns
    -------
    thresh_array: np.ndarray
        The segmented image array.
    """

    if Apply_Mask and Mask is not None:
        eval_array: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        eval_array = imarray

    thresh_vals: Any = segment_util.calc_thresholds(
        eval_array, Method, Otsu_Classes, Bin_Count, Max_Iteration_Count
    )

    thresh_array: np.ndarray = np.zeros(imarray.shape, np.uint8)
    if isinstance(thresh_vals, np.ndarray):
        intensities: np.ndarray = np.linspace(
            0, 255, len(thresh_vals) + 1, dtype=np.uint8
        )
        for index, thresh in enumerate(thresh_vals, start=1):
            thresh_array[imarray > thresh] = intensities[index]
    else:
        thresh_array[imarray > thresh_vals] = 255

    return thresh_array


@register_operation(category="segmentation", display_name="Hysteresis Threshold")
def hysteresis(
    imarray: np.ndarray,
    Thresholds: tuple[
        Annotated[float, {"min": -32768, "max": 65535}],
        Annotated[float, {"min": -32768, "max": 65535}],
    ] = (0, 0),
) -> np.ndarray:
    """
    Segment an image using the hysteresis method. Values below the lower threshold are
    segemented only if they are attached to a region greater than the upper threshold.
    Convenience wrapper around scikit-image's filters.apply_hysteresis_threshold:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.apply_hysteresis_threshold

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be segmented.
    Thresholds: tuple
        The lower and upper thresholds to use for hysteresis thresholding.

    Returns
    -------
    thresh_array: np.ndarray
        The segmented image array.
    """

    return util.convert_dtype(
        apply_hysteresis_threshold(imarray, min(Thresholds), max(Thresholds)), "Uint8"
    )
