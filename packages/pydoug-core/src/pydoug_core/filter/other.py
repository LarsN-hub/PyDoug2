"""
Other miscellaneous filters.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["entropy", "noise"]


@register_operation(category="filter", display_name="Entropy")
def entropy(
    imarray: np.ndarray,
    Radius: Annotated[int, {"min": 1}] = 3,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Find the local entropy of an image. Good for highlighting areas with different
    grayscale complexities. Can handle 2D/3D but is performed channelwise for RGB.
    Requires integer format dtype. Convenience wrapper around scikit-image's
    filters.rank.entropy:
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.entropy

    Parameters
    ----------
    imarray: np.ndarray
        The image to filter.
    Radius: int
        The radius of the local footprint.
    Perform_Slicewise: bool
        Whether or not to filter the image slicewise (True, requires 3D) or in bulk
        (False). Deafult is False.

    Returns
    -------
    entropy_array: np.ndarray
        The image after filtering.
    """

    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import ball, disk

    from pydoug_core.util import is3d, isrgb

    if is3d(imarray) and not Perform_Slicewise:
        footprint = ball(Radius)
    else:
        footprint = disk(Radius)

    if Perform_Slicewise and is3d(imarray):
        entropy_array: np.ndarray = np.empty(imarray.shape)
        if isrgb(imarray):
            for cindex in range(0, imarray.shape[-1]):
                for index in range(0, imarray.shape[0]):
                    entropy_array[index, :, :, cindex] = rank_entropy(
                        imarray[index, :, :, cindex], footprint
                    )
        else:
            for index in range(0, imarray.shape[0]):
                entropy_array[index, ...] = rank_entropy(imarray[index, ...], footprint)

        return entropy_array

    if isrgb(imarray):
        entropy_array: np.ndarray = np.empty(imarray.shape)
        for cindex in range(0, imarray.shape[-1]):
            entropy_array[..., cindex] = rank_entropy(imarray[..., cindex], footprint)

        return entropy_array

    return rank_entropy(imarray, footprint)


@register_operation(category="filter", display_name="Add Noise")
def noise(
    imarray: np.ndarray,
    Method: Literal[
        "Gaussian", "Poisson", "Salt", "Pepper", "S&P", "Speckle"
    ] = "Gaussian",
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Add random noise to an image. Convenience wrapper around scikit-image's
    util.random_noise:
    https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise

    Parameters
    ----------
    imarray: np.ndarray
        The image to add noise to.
    Method: Literal
        The method to use for adding noise to the image. Default is "Gaussian".
        - "Gaussian": Adds Gaussian-distributed additive noise.
        - "Poisson": Adds Poisson-distributed noise generated from the image's values.
        - "Salt": Replaces random pixels with the brightest value in the dtype.
        - "Pepper": Replaces random pixels with the darkest value in the dtype.
        - "S&P": Combines "Salt" and "Pepper" methods.
        - "Speckle": Multiplicative Gaussian-distributed noise.
    Preserve_Dtype: bool
        Whether to preserve the image's dtype (True) or not (False). Default is True.

    Returns
    -------
    noise_array: np.ndarray
        The image with noise added.
    """

    from skimage.util import random_noise

    from pydoug_core.util import convert_dtype, dtype_2_str

    method = Method.lower()
    if Method == "S&P":
        method = "s&p"
    noise_array: np.ndarray = random_noise(imarray, method)

    if Preserve_Dtype:
        return convert_dtype(noise_array, dtype_2_str(imarray.dtype))  # type: ignore

    return noise_array
