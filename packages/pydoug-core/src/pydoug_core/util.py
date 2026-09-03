"""
Miscellaneous utility functions for PyDoug core package.
"""

import numpy as np
from skimage.color import rgb2gray as skrgb2gray

from pydoug_core.registry import register_operation

__all__ = [
    "is3d",
    "isrgb",
    "gray2rgb",
    "rgb2gray",
    "chan_axis",
    "output_dtype",
    "axlengths",
]


def is3d(imarray: np.ndarray | None = None, *, imshape: None | tuple = None) -> bool:
    """
    Accept an image array or its shape and return if it is 3D or not.
    Assumes 3rd or 4th axis with size 3 indicates an RGB array.

    Parameters
    ----------
    imarray: np.ndarray | None
        Image array. Optional if shape is provided.
    imshape: None | tuple
        Shape of an image array. Optional if imarray is provided.

    Returns
    -------
    True if image array/shape is 3D, False if not.
    """

    if not imshape:
        if isinstance(imarray, np.ndarray):
            imshape = imarray.shape
        else:
            return False
    if isinstance(imshape, tuple):
        if len(imshape) < 3:
            return False
        elif len(imshape) == 3:
            if imshape[2] == 3:
                return False
            else:
                return True
        elif len(imshape) > 3:
            return True


def isrgb(imarray: np.ndarray | None = None, imshape: None | tuple = None) -> bool:
    """
    Accept an image array or its shape and return if it is RGB or not.
    Assumes 3rd or 4th axis with size 3 indicates an RGB array.

    Parameters
    ----------
    imarray: np.ndarray | None
        Image array. Optional if shape is provided.
    imshape: None | tuple
        Shape of an image array. Optional if imarray is provided.

    Returns
    -------
    True if image array/shape is RGB, False if not.
    """

    if not imshape:
        if isinstance(imarray, np.ndarray):
            imshape = imarray.shape
        else:
            return False
    if isinstance(imshape, tuple):
        check3d: bool = is3d(imshape=imshape)
        if not check3d and len(imshape) > 2:
            return True
        elif check3d and len(imshape) > 3:
            return True
        else:
            return False


@register_operation(category="utilities", display_name="Grayscale to RGB")
def gray2rgb(imarray: np.ndarray) -> np.ndarray:
    """
    Make a grayscale image array an RGB image array.

    Parameters
    ----------
    imarray: np.ndarray
        The grayscale image array.

    Returns
    -------
    rgbarray: np.ndarray
        The RGB image version of imarray.
        If isrgb(imarray) is True, returns the original array.
    """

    if isrgb(imarray):
        return imarray
    else:
        return np.repeat(np.expand_dims(imarray, imarray.ndim), 3, imarray.ndim)


@register_operation(category="utilities", display_name="RGB to Grayscale")
def rgb2gray(imarray: np.ndarray) -> np.ndarray:
    """
    Make an RGB image array a grayscale image array.
    """

    if isrgb(imarray):
        return skrgb2gray(imarray)
    return imarray


def chan_axis(imarray: np.ndarray) -> int | None:
    """
    Return the channel axis of an RGB array. Returns None for a grayscale array.

    Parameteres
    -----------
    imarray: np.ndarray
        The image array to obtain the channel axis for.

    Returns
    -------
    ax: int | None
        The channel axis of an RGB image array or None for a grayscale array.
    """

    if isrgb(imarray):
        return imarray.ndim - 1
    else:
        return None


def output_dtype(
    imarray: np.ndarray,
    conv_dtype: np.dtype = np.dtype(np.float64),
    preserve_dtype: bool = True,
) -> np.dtype:
    """
    Return the required output dtype of a function where conversion is necessary.

    Parameters
    ----------
    imarray: np.ndarray
        The image array being converted.
    preserve_dtype: bool
        Whether or not to preserve the input dtype. Default is true.

    Returns
    -------
    outdtype: np.dtype
        The required output dtype for the function.
    """

    if preserve_dtype:
        return imarray.dtype
    return conv_dtype


def axlengths(imarray: np.ndarray) -> dict[str, int | None]:
    """
    Return the axial lengths of an image array in a dictionary.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to extract axial lengths from.

    Returns
    -------
    axlengths: dict[str, int | None]
        A dictionary containing the axial lengths in the format "x": 50. Always returns
        inegers for "x" and "y" but will return "None" for "z" in case of a 2D array.
    """

    if is3d(imarray):
        return {"x": imarray.shape[2], "y": imarray.shape[1], "z": imarray.shape[0]}
    else:
        return {"x": imarray.shape[1], "y": imarray.shape[0], "z": None}
