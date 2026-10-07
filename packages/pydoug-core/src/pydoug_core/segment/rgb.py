"""
Methods for segmenting and labeling RGB images.
"""

from typing import Annotated

import numpy as np
from skimage.segmentation import felzenszwalb, quickshift, slic

from pydoug_core.registry import register_operation

__all__ = ["fz", "qs", "sl"]


@register_operation(category="segment", display_name="Felzenszwalb")
def fz(
    imarray: np.ndarray,
    Scale: Annotated[float, {"min": 0}] = 1,
    Sigma: Annotated[float, {"min": 0}] = 0.8,
    Minimum_Size: Annotated[int, {"min": 0}] = 20,
) -> np.ndarray:
    """
    Segment and label a 2D image with Felzenswalb's algorithm (can segment RGB images).
    Convenience wrapper around scikit-image's segmentation.felzenswalb:
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Scale: float
        The scale parameter of the algorithm. Higher scale = larger segments.
    Sigma: float
        The standard deviation of the Gaussian kernel for preprocessing the image.
    Minimum_Size: int
        The minimum size of a segment.

    Returns
    -------
    seg_array: np.ndarray
        The segmented image.
    """

    return felzenszwalb(imarray, Scale, Sigma, Minimum_Size)


@register_operation(category="segment", display_name="Quickshift")
def qs(
    imarray: np.ndarray,
    Ratio: Annotated[float, {"min": 0, "max": 1}] = 1.0,
    Kernel_Size: Annotated[int, {"min": 3}] = 5,
    Maximum_Distance: Annotated[int, {"min": 1}] = 10,
    Sigma: Annotated[int, {"min": 0}] = 0,
) -> np.ndarray:
    """
    Segment and label a 2D image with the quickshift algorithm (can segment RGB images).
    Convenience wrapper around scikit-image's segment.quickshift:
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift

    Parameters
    ----------
    imarray: np.ndarray
        The image to segment.
    Ratio: float
        Ratio in the range [0, 1] to balance color vs. image space. Higher = more weight
        for color space.
    Kernel_Size: int
        Width of the Gaussian kernel used to smooth sample density.
        Higher = larger segments.
    Maximum_Distance: int
        The maximum distance for separating segments. Higher = larger segments.
    Sigma: int
        The standard deviation of the Gaussian kernel for preprocessing the image.

    Returns
    -------
    seg_array: np.ndarray
        The segmented image.
    """

    return quickshift(imarray, Ratio, Kernel_Size, Maximum_Distance, sigma=Sigma)


@register_operation(category="segment", display_name="SLIC")
def sl(
    imarray: np.ndarray,
    Label_Count: Annotated[int, {"min": 0}] = 100,
    Compactness: Annotated[float, {"min": 0}] = 10,
    Maximum_Iterations: Annotated[int, {"min": 0}] = 10,
    Sigma: Annotated[float, {"min": 0}] = 0,
    Size_Factors: tuple[
        Annotated[float, {"min": 0}],
        Annotated[float, {"min": 0}],
    ] = (0.5, 3),
) -> np.ndarray:
    """
    Segment and label an image with the SLIC algorithm (can segment RGB images, 2/3D).
    Convenience wrapper around scikit-image's segment.slic:
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic

    Parameters
    ----------
    imarray: np.ndarray
        The image to segment.
    Label_Count: int
        The number of labels to include in segmented and labeled image.
    Compactness: float
        Balances color and space proximity. Higher = more weight to space.
    Maximum_Iterations: int
        The maximum number of iterations of k-means clustering.
    Sigma: float
        The standard deviation of the Gaussian kernel for preprocessing the image.
    Size_Factors: tuple
        The minimum and maximum size factors for the labeled segments. The minimum is
        used as the proportion to remove the smallest segmented. The maximum is used as
        the proportion of the maximum connected segment.

    Returns
    -------
    seg_array: np.ndarray
        The segmented image.
    """

    return slic(
        imarray,
        Label_Count,
        Compactness,
        Maximum_Iterations,
        Sigma,
        min_size_factor=min(Size_Factors),
        max_size_factor=max(Size_Factors),
    )
