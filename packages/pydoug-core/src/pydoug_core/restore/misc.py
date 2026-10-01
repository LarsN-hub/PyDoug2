"""
Miscellaneous restoration filters.
"""

from typing import Annotated

import numpy as np
from skimage.restoration import inpaint_biharmonic, rolling_ball

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["rolling", "inpaint"]


@register_operation(category="restore", display_name="Rolling Ball")
def rolling(
    imarray: np.ndarray,
    Radius: Annotated[int, {"min": 1}] = 5,
    Return_Background: bool = False,
) -> np.ndarray:
    """
    Use a rolling ball algorithm to attempt to subtract the background of an image.
    The background must be darker than the foreground.

    Parameters
    ----------
    imarray: np.ndarray
        The image to perform the rolling ball algorithm on.
    Radius: int
        The radius of the ball.
    Return_Background: bool
        Whether to return the background image (True) or the image with the background
        subtracted (False). Default is False.

    Returns
    -------
    sub_array: np.ndarray
        The image with the background subtracted.
    """

    bg: np.ndarray = rolling_ball(imarray, radius=Radius)
    if Return_Background:
        return bg
    return imarray - bg


@register_operation(
    category="restore",
    display_name="Inpaint",
    array_params=("Mask",),
)
def inpaint(imarray: np.ndarray, Mask: np.ndarray) -> np.ndarray:
    """
    Inpaint masked values of an image. Good for removing small specs from dust on a
    camera lens, etc. Convenience wrapper around scikit-image's
    restore.inpaint_biharmonic:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.inpaint_biharmonic

    Parameters
    ----------
    imarray: np.ndarray
        The image to be deconvoluted.
    Mask: np.ndarray
        The mask defining values to be inpainted. If "Mask" is not binary it will be
        converted before use.

    Returns
    -------
    inpaint_array: np.ndarray
        The inpainted image.
    """

    return inpaint_biharmonic(imarray, util.prep_mask(imarray, Mask)).astype(
        imarray.dtype
    )
