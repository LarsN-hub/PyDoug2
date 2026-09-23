"""
Local intensity-based thresholding operations to segment images.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola
from skimage.filters.rank import otsu as rank_otsu
from skimage.filters.rank import threshold as rank_thresh
from skimage.morphology import ball, disk

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["local_threshold"]


@register_operation(category="segmentation", display_name="Local Threshold")
def local_threshold(
    imarray: np.ndarray,
    Method: Literal["Adaptive", "Mean", "Niblack", "Sauvola", "Otsu"] = "Otsu",
    Radius: Annotated[int, {"min": 1}] = 5,
    k: Annotated[float, {"min": 0}] = 0.2,
    Adaptive_Method: Literal["Gaussian", "Generic", "Mean", "Median"] = "Gaussian",
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
) -> np.ndarray:
    """
    Segment an image using local adaptive algorithms. Convenience wrapper around the
    local threshold functions in scikit-image's filters and filters.rank packages:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_niblack
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.otsu
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.threshold

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Method: Literal["Adaptive", "Mean", "Niblack", "Sauvola", "Otsu"]
        The local algorithm to use to segment the image. Default is Otsu.
    Radius: int
        The radius for the local footprint. Must be odd if "Method" is
        "Adaptive", "Niblack", or "Sauvola".
    k: float
        If "Method" is "Niblack" or "Sauvola", the k parameter for the calculation.
        Ignored for other methods.
    Adaptive_Method: Literal["Gaussian", "Generic", "Mean", "Median"]
        If "Method" is "Adaptive", the method used to calculate the local threshold.
        Default is "Gaussian".
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        If "Method" is "Adaptive", how the algorithm should handle image borders.
        Default is "Nearest". Ignroed for other methods.
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.

    Returns
    -------
    thresh_array: np.ndarray
        The segmented image array.
    """

    if Method == "Adaptive":
        thresh_mask: np.ndarray = threshold_local(
            imarray,
            Radius,
            Adaptive_Method.lower(),
            mode=Edge_Mode.lower(),
            cval=Constant_Value,
        )

    if Method == "Mean":
        if util.is3d(imarray):
            footprint = ball(Radius)
        else:
            footprint = disk(Radius)
        return util.convert_dtype(
            rank_thresh(imarray, footprint).astype(np.bool), "Uint8"
        )

    if Method == "Niblack":
        thresh_mask: np.ndarray = threshold_niblack(imarray, Radius, k)

    if Method == "Otsu":
        if util.is3d(imarray):
            footprint = ball(Radius)
        else:
            footprint = disk(Radius)
        thresh_mask: np.ndarray = rank_otsu(imarray, footprint)

    if Method == "Sauvola":
        thresh_mask: np.ndarray = threshold_sauvola(imarray, Radius, k)

    return util.convert_dtype(imarray > thresh_mask, "Uint8")  # type: ignore
