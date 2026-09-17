"""
Thresholding operations to segment images.
"""

from typing import Annotated

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["threshold_array"]


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

    thresh_array: np.ndarray
        The thresholded image array.
    """

    return threshold_array(imarray, Range)
