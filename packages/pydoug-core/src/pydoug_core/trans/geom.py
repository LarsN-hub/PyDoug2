"""
Geometric transformations of images.
"""

from typing import cast

import numpy as np
from skimage import transform

from pydoug_core import util

__all__ = ["mirror", "reslice", "rotate", "rescale", "resize"]


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

    outdtype: np.dtype = util.output_dtype(imarray, preserve_dtype=preserve_dtype)

    if clockwise:
        angle *= -1

    def _3drot(imarray: np.ndarray, angle: float, resize: bool) -> np.ndarray:
        return np.moveaxis(
            transform.rotate(
                np.moveaxis(imarray, 0, 2), angle, resize, preserve_range=True
            ),
            2,
            0,
        )

    if util.is3d(imarray):
        if util.isrgb(imarray):
            return np.stack(
                (
                    _3drot(imarray[:, :, :, 0], angle, resize),
                    _3drot(imarray[:, :, :, 1], angle, resize),
                    _3drot(imarray[:, :, :, 2], angle, resize),
                ),
                axis=3,
            )
        else:
            return _3drot(imarray, angle, resize).astype(outdtype, copy=False)
    else:
        return transform.rotate(imarray, angle, resize, preserve_range=True).astype(
            outdtype, copy=False
        )


def rescale(
    imarray: np.ndarray,
    scale: float = 1,
    order: int = 1,
    preserve_dtype: bool = True,
    segmented_data: bool = False,
) -> np.ndarray:
    """
    Rescale an image array by a scale factor.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be rescaled.
    scale: float
        The factor to rescale the image by.
    order: int
        The order (0-5) to use for interpolation. If segmented_data is False,
        order will be forced to 0. Default is the bilinear (1).
    preserve_dtype: bool
        Whether or not to return the same data type that was input.
        Default is True (converts np.float64 rotation results to original data type).
    segmented_data: bool
        Whether the input data is segmented (True) or continuous (False). If true,
        anti-aliasing will be avoided during downsampling to preserve values.
        Default is False (anti-aliases when downsampling non-boolean arrays).

    Returns
    -------
    resarray: np.ndarray
        The rescaled image array.
    """

    if scale == 1:
        return imarray

    outdtype: np.dtype = util.output_dtype(imarray, preserve_dtype=preserve_dtype)
    if not segmented_data and imarray.dtype != np.dtype(np.bool):
        antialias: bool = True
    else:
        antialias = False
        order = 0

    return transform.rescale(
        imarray,
        scale,
        order,
        preserve_range=True,
        anti_aliasing=antialias,
        channel_axis=util.chan_axis(imarray),
    ).astype(outdtype, copy=False)


def resize(
    imarray: np.ndarray,
    xshape: int | None = None,
    yshape: int | None = None,
    zshape: int | None = None,
    order: int = 1,
    preserve_dtype: bool = True,
    segmented_data: bool = False,
) -> np.ndarray:
    """
    Resize an image array to specified dimensions.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be resized.
    _shape: int | None
        The length of the output image array along each axis (_ = x, y, or z). If not
        provided, the length of the input image array along that axis will be used.
    order: int
        The order (0-5) to use for interpolation. If segmented_data is False,
        order will be forced to 0. Default is the bilinear (1).
    preserve_dtype: bool
        Whether or not to return the same data type that was input.
        Default is True (converts np.float64 rotation results to original data type).
    segmented_data: bool
        Whether the input data is segmented (True) or continuous (False). If true,
        anti-aliasing will be avoided during downsampling to preserve values.
        Default is False (anti-aliases when downsampling non-boolean arrays).

    Returns
    -------
    resarray: np.ndarray
        The resized image array.
    """

    axlengths: dict[str, int | None] = {"x": xshape, "y": yshape, "z": zshape}
    if axlengths == util.axlengths(imarray):
        return imarray
    if xshape is None and yshape is None and zshape is None:
        return imarray

    outshape: list = []
    for key in list(axlengths.keys()):
        if axlengths[key] is None:
            axlengths[key] = util.axlengths(imarray)[key]
    if util.is3d(imarray):
        for key in ["z", "y", "x"]:
            outshape.append(axlengths[key])
    else:
        for key in ["y", "x"]:
            outshape.append(axlengths[key])
    if util.isrgb(imarray):
        outshape.append(imarray.shape[-1])

    outdtype: np.dtype = util.output_dtype(imarray, preserve_dtype=preserve_dtype)
    if not segmented_data and imarray.dtype != np.dtype(np.bool):
        antialias: bool = True
    else:
        antialias = False
        order = 0

    resarray = transform.resize(
        imarray, outshape, order, preserve_range=True, anti_aliasing=antialias
    )
    return cast(np.ndarray, resarray).astype(outdtype, copy=False)
