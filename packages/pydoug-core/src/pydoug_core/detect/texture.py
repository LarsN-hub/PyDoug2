"""
Texture-based features.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation
from pydoug_core.util import is3d, isrgb

__all__ = ["local_binary_pattern"]


@register_operation(category="detect", display_name="Local Binary Pattern")
def local_binary_pattern(
    imarray: np.ndarray,
    Method: Literal["Default", "NRI Uniform", "ROR", "Uniform", "Variance"] = "Default",
    Number_of_Points: Annotated[int, {"min": 1}] = 25,
    Radius: Annotated[int, {"min": 1}] = 3,
) -> np.ndarray:
    """
    Find the local binary pattern of a grayscale image. Performed slicewise for 3D.
    Good for classifying textures. Convenience wrapper around scikit-image's
    feature.local_binary_pattern:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern

    Parameters
    ----------
    imarray: np.ndarray
        The image to find the local binary pattern for.
    Method: Literal["Default", "NRI Uniform", "ROR", "Uniform", "Variance"]
        The method for finding the local binary pattern. Default is "Default".
        - "Default": Original method. Grayscale invariant but not rotation invariant.
        - "NRI Uniform": Variant of "Uniform" that is not rotation invariant.
        - "ROR": Extends "Default" to be rotation invariant.
        - "Uniform": Uniform pattern. Grayscale and rotation invariant.
        - "Variance": Texture variance. Rotation invariant but not grayscale invariant.
    Number_of_Points: int
        The number of neighbor set points to compute the local binary pattern.
    Radius: int
        The radius of the local kernel.

    Returns
    -------
    lbp_array: np.ndarray
        The local binary pattern of the image.
    """

    if isrgb(imarray):
        print("Image is RGB, local binary pattern not calculated!")
        return imarray

    from skimage.feature import local_binary_pattern as lbp

    def _lbp(imarray, Method, Number_of_Points, Radius) -> np.ndarray:
        if Method == "Default":
            Method = "default"
        elif Method == "NRI Uniform":
            Method = "nri_uniform"
        elif Method == "ROR":
            Method = "ror"
        elif Method == "Uniform":
            Method = "uniform"
        else:
            Method = "var"

        return lbp(imarray, Number_of_Points, Radius, Method)

    if is3d(imarray):
        lbp_array: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            lbp_array[index] = _lbp(imarray[index], Method, Number_of_Points, Radius)

        return lbp_array

    return _lbp(imarray, Method, Number_of_Points, Radius)
