"""
Detect features in images like corners, blobs, and edges.
"""

from typing import Annotated

import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import ball, disk

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["peaks"]


@register_operation(category="features", display_name="Peaks")
def peaks(
    imarray: np.ndarray,
    Radius: Annotated[int, {"min": 0}] = 0,
    Minimum_Distance: Annotated[int, {"min": 1}] = 1,
    Absolute_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0,
    Relative_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Identify local intensity maxima in an image. Can handle 2D/3D grayscale/RGB images.
    Convenience wrapper around scikit-image's feature.peak_local_max:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max

    Parameters
    ----------
    imarray: np.ndarray
        The image to detect peaks in.
    Radius: int
        (Optional) The radius of the local footprint to find local maxima within. If 0,
        automatically generated.
    Minimum_Distance: int
        The minimum distance between peaks.
    Absolute_Threshold: float
        The abolsute minimum intensity of peaks. If 0, will be calculated automaticall.
    Relative_Threshold: float
        The relative minimum intensity of peaks (found as this value * image max).
        If 0, will be calculated automatically. Overwrites "Absolute_Threshold" if both
        are provided.
    Perform_Slicewise: boo
        Whether or not to perform the operation slicewise. Default is False.

    Returns
    -------
    peak_array: np.ndarray
        A binary image where True values are local peaks in the original image.
    """

    abs_thresh: float | None = Absolute_Threshold
    if abs_thresh == 0:
        abs_thresh = None
    rel_thresh: float | None = Relative_Threshold
    if rel_thresh == 0:
        rel_thresh = None
    else:
        abs_thresh = None

    if Radius == 0:
        footprint = None
    else:
        if Perform_Slicewise or not util.is3d(imarray):
            footprint = disk(Radius)
        else:
            footprint = ball(Radius)

    if util.isrgb(imarray):
        peak_array: np.ndarray = np.zeros(imarray.shape[:-1], np.bool)
    else:
        peak_array: np.ndarray = np.zeros(imarray.shape, np.bool)
    if Perform_Slicewise and util.is3d(imarray):
        for index in range(0, imarray.shape[0]):
            peak_coords = peak_local_max(
                imarray[index],
                Minimum_Distance,
                abs_thresh,
                rel_thresh,
                footprint=footprint,
            )
            if util.isrgb(imarray):
                peak_coords = peak_coords[:, :-1]
            peak_array[index][tuple(peak_coords.T)] = True

    peak_coords: np.ndarray = peak_local_max(
        imarray, Minimum_Distance, abs_thresh, rel_thresh, footprint=footprint
    )
    if util.isrgb(imarray):
        peak_coords = peak_coords[:, :-1]
    peak_array[tuple(peak_coords.T)] = True
    return peak_array
