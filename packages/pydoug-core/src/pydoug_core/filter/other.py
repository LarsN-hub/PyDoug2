"""
Other miscellaneous filters.
"""

from typing import Annotated

import numpy as np
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import ball, disk

from pydoug_core.registry import register_operation
from pydoug_core.util import is3d, isrgb

__all__ = ["entropy"]


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
