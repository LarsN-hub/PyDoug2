"""
Image transformations.
"""

import numpy as np
from skimage import transform

from pydoug_core import util


def mirror(imarray: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Mirror an image array across an axis.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to mirror.
    axis: int
        The axis to mirror across.

    Returns
    -------
    mirarray: np.ndarray
        The mirrored image array.
    """

    if not util.is3d(imarray) and axis >= 2:
        return imarray

    return np.flip(imarray, axis)


def reslice(imarray: np.ndarray, orientation: str = "top") -> np.ndarray:
    """
    Reslice a 3D image array.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to reslice.
    orientation: str
        The orientation of the array after reslicing.

    Returns
    -------
    resarray: np.ndarray
        The resliced image array.
    """

    if not util.is3d(imarray):
        return imarray

    if orientation == "top":
        return mirror(np.moveaxis(imarray, 1, 0), 1)
    elif orientation == "bottom":
        return mirror(np.moveaxis(imarray, 1, 0), 0)
    elif orientation == "left":
        return mirror(np.swapaxes(imarray, 2, 0), 2)
    elif orientation == "right":
        return mirror(np.swapaxes(imarray, 2, 0), 0)
    elif orientation == "back":
        return mirror(mirror(imarray, 0), 2)
    else:
        return imarray


def rotate(
    imarray: np.ndarray,
    angle: float = 0,
    clockwise: bool = False,
    resize: bool = False,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """
    Rotate an image around its forward-facing central axis.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be rotated.
    angle: float
        The angle to rotate by.
    clockwise: bool
        Whether or not to rotate clockwise. Default is False (counter clockwise).
    resize: bool
        Whether or not to resize the array to accomodate the rotation.
        Default is False (dimensions are maintained).
    preserve_dtype: bool
        Whether or not to return the same data type that was input.
        Default is True (converts np.float64 rotation results to original data type).

    Returns
    -------
    rotarray: np.ndarray
        The rotated image array.
    """

    if angle == 0 or angle == 360:
        return imarray

    if preserve_dtype:
        outdtype: np.dtype = imarray.dtype
    else:
        outdtype: np.dtype = np.dtype(np.float64)
    if clockwise:
        angle *= -1

    return transform.rotate(imarray, angle, resize, preserve_range=True).astype(
        outdtype, copy=False
    )
