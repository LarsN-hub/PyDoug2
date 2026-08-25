"""
Miscellaneous utility functions for PyDoug core package.
"""

import numpy as np


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
