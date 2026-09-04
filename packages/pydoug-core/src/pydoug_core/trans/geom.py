"""
Geometric transformations of images.
"""

from typing import cast

import numpy as np
from skimage import transform

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["mirror", "reslice", "rotate", "rescale", "resize"]


@register_operation(category="transformations", display_name="Mirror")
def mirror(imarray: np.ndarray, Axis: str = "Z") -> np.ndarray:
    """
    Mirror an image array across an axis.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to mirror.
    Axis: str
        The axis to mirror across.

    Returns
    -------
    mirarray: np.ndarray
        The mirrored image array.
    """

    axis: int = util.axstr2int(imarray, Axis)
    if not util.is3d(imarray) and axis >= 2:
        return imarray

    return np.flip(imarray, axis)


@register_operation(category="transformations", display_name="Reslice")
def reslice(imarray: np.ndarray, Orientation: str = "top") -> np.ndarray:
    """
    Reslice a 3D image array.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to reslice.
    Orientation: str
        The orientation of the array after reslicing.

    Returns
    -------
    resarray: np.ndarray
        The resliced image array.
    """

    if not util.is3d(imarray):
        return imarray

    if Orientation == "top":
        return mirror(np.moveaxis(imarray, 1, 0), util.axint2str(imarray, 1))
    elif Orientation == "bottom":
        return mirror(np.moveaxis(imarray, 1, 0), util.axint2str(imarray, 0))
    elif Orientation == "left":
        return mirror(np.swapaxes(imarray, 2, 0), util.axint2str(imarray, 2))
    elif Orientation == "right":
        return mirror(np.swapaxes(imarray, 2, 0), util.axint2str(imarray, 0))
    elif Orientation == "back":
        return mirror(
            mirror(imarray, util.axint2str(imarray, 0)), util.axint2str(imarray, 2)
        )
    else:
        return imarray


@register_operation(category="transformations", display_name="Rotate")
def rotate(
    imarray: np.ndarray,
    Angle: float = 0,
    Clockwise: bool = False,
    Resize: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Rotate an image around its forward-facing central axis.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be rotated.
    Angle: float
        The angle to rotate by.
    clockwise: bool
        Whether or not to rotate clockwise. Default is False (counter clockwise).
    Resize: bool
        Whether or not to resize the array to accomodate the rotation.
        Default is False (dimensions are maintained).
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input.
        Default is True (converts np.float64 rotation results to original data type).

    Returns
    -------
    rotarray: np.ndarray
        The rotated image array.
    """

    if Angle == 0 or Angle == 360:
        return imarray

    outdtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)

    if Clockwise:
        Angle *= -1

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
                    _3drot(imarray[:, :, :, 0], Angle, Resize),
                    _3drot(imarray[:, :, :, 1], Angle, Resize),
                    _3drot(imarray[:, :, :, 2], Angle, Resize),
                ),
                axis=3,
            )
        else:
            return _3drot(imarray, Angle, Resize).astype(outdtype, copy=False)
    else:
        return transform.rotate(imarray, Angle, Resize, preserve_range=True).astype(
            outdtype, copy=False
        )


@register_operation(category="transformations", display_name="Rescale")
def rescale(
    imarray: np.ndarray,
    Scale: float = 1,
    Order: int = 1,
    Segmented_Data: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Rescale an image array by a scale factor.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be rescaled.
    Scale: float
        The factor to rescale the image by.
    Order: int
        The order (0-5) to use for interpolation. If segmented_data is False,
        order will be forced to 0. Default is the bilinear (1).
    Segmented_Data: bool
        Whether the input data is segmented (True) or continuous (False). If true,
        anti-aliasing will be avoided during downsampling to preserve values.
        Default is False (anti-aliases when downsampling non-boolean arrays).
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input.
        Default is True (converts np.float64 rotation results to original data type).

    Returns
    -------
    resarray: np.ndarray
        The rescaled image array.
    """

    if Scale == 1:
        return imarray

    outdtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)
    if not Segmented_Data and imarray.dtype != np.dtype(np.bool):
        antialias: bool = True
    else:
        antialias = False
        Order = 0

    return transform.rescale(
        imarray,
        Scale,
        Order,
        preserve_range=True,
        anti_aliasing=antialias,
        channel_axis=util.chan_axis(imarray),
    ).astype(outdtype, copy=False)


@register_operation(category="transformations", display_name="Resize")
def resize(
    imarray: np.ndarray,
    X_Length: int | None = None,
    Y_Length: int | None = None,
    Z_Length: int | None = None,
    Order: int = 1,
    Segmented_Data: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Resize an image array to specified dimensions.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be resized.
    __Length: int | None
        The length of the output image array along each axis (__ = X_, Y_, or Z_). If
        not provided, the length of the input image array along that axis will be used.
    Order: int
        The order (0-5) to use for interpolation. If Segmented_Data is False,
        order will be forced to 0. Default is the bilinear (1).
    Segmented_Data: bool
        Whether the input data is segmented (True) or continuous (False). If true,
        anti-aliasing will be avoided during downsampling to preserve values.
        Default is False (anti-aliases when downsampling non-boolean arrays).
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input.
        Default is True (converts np.float64 rotation results to original data type).

    Returns
    -------
    resarray: np.ndarray
        The resized image array.
    """

    axlengths: dict[str, int | None] = {"x": X_Length, "y": Y_Length, "z": Z_Length}
    if axlengths == util.axlengths(imarray):
        return imarray
    if X_Length is None and Y_Length is None and Z_Length is None:
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

    outdtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)
    if not Segmented_Data and imarray.dtype != np.dtype(np.bool):
        antialias: bool = True
    else:
        antialias = False
        Order = 0

    resarray = transform.resize(
        imarray, outshape, Order, preserve_range=True, anti_aliasing=antialias
    )
    return cast(np.ndarray, resarray).astype(outdtype, copy=False)
