"""
Geometric transformations of images.
"""

from typing import Literal, cast

import numpy as np
from skimage import transform

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["mirror", "reslice", "rotate", "rescale", "resize"]


@register_operation(category="transformations", display_name="Mirror")
def mirror(
    imarray: np.ndarray, Direction: Literal["X", "Y", "Z", "C"] = "Z"
) -> np.ndarray:
    """
    Mirror an image array across an axis. Convenience wrapper around numpy's "flip":
    https://numpy.org/doc/stable/reference/generated/numpy.flip.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to mirror.
    Direction: Literal["X", "Y", "Z", "C"]
        The direction to mirror along.

    Returns
    -------
    mirarray: np.ndarray
        The mirrored image array.
    """

    axis: int = util.axstr2int(imarray, Direction)
    if not util.is3d(imarray) and axis >= 2:
        return imarray

    return np.flip(imarray, axis)


@register_operation(category="transformations", display_name="Reslice")
def reslice(
    imarray: np.ndarray,
    Orientation: Literal["Top", "Bottom", "Left", "Right", "Back"] = "Top",
) -> np.ndarray:
    """
    Reslice a 3D image array. Convenience wrapper around numpy's "moveaxis" and
    "swapaxes" functions:
    https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html
    https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to reslice.
    Orientation: Literal["Top", "Bottom", "Left", "Right", "Back"]
        The orientation of the array after reslicing.

    Returns
    -------
    resarray: np.ndarray
        The resliced image array.
    """

    if not util.is3d(imarray):
        return imarray

    if Orientation == "Top":
        return mirror(np.moveaxis(imarray, 1, 0), util.axint2str(imarray, 1))
    elif Orientation == "Bottom":
        return mirror(np.moveaxis(imarray, 1, 0), util.axint2str(imarray, 0))
    elif Orientation == "Left":
        return mirror(np.swapaxes(imarray, 2, 0), util.axint2str(imarray, 2))
    elif Orientation == "Right":
        return mirror(np.swapaxes(imarray, 2, 0), util.axint2str(imarray, 0))
    elif Orientation == "Back":
        return mirror(
            mirror(imarray, util.axint2str(imarray, 0)), util.axint2str(imarray, 2)
        )
    else:
        return imarray


@register_operation(
    category="transformations",
    display_name="Rotate",
    widget_options={"Angle": {"max": 360}},
)
def rotate(
    imarray: np.ndarray,
    Angle: float = 0,
    Clockwise: bool = False,
    Resize: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Rotate an image around its forward-facing central axis. Convenience wrapper around
    scikit-image's transform.rotate:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rotate

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be rotated.
    Angle: float
        The angle to rotate by.
    Clockwise: bool
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


@register_operation(
    category="transformations",
    display_name="Rescale",
    widget_options={"Scale": {"step": 0.001, "min": 0.001}},
    native_options={"Scale": {"setDecimals": 3}},
)
def rescale(
    imarray: np.ndarray,
    Scale: float = 1,
    Order: Literal[0, 1, 2, 3, 4, 5] = 1,
    Segmented_Data: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Rescale an image array by a scale factor. Convenience wrapper around scikit-image's
    transform.rescale function:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rescale

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be rescaled.
    Scale: float
        The factor to rescale the image by.
    Order: Literal[0, 1, 2, 3, 4, 5] = 1
        The order (0-5) to use for interpolation. If Segmented Data is False,
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


@register_operation(
    category="transformations",
    display_name="Resize",
    widget_options={
        "X_Length": {"max": 10000},
        "Y_Length": {"max": 10000},
        "Z_Length": {"max": 10000},
    },
)
def resize(
    imarray: np.ndarray,
    X_Length: int | None = None,
    Y_Length: int | None = None,
    Z_Length: int | None = None,
    Order: Literal[0, 1, 2, 3, 4, 5] = 1,
    Segmented_Data: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Resize an image array to specified dimensions. Convenience wrapper around
    scikit-image's transform.resize function:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be resized.
    X_Length: int | None
        The length of the output image array along the x-axis. If not provided, the
        length of the input image array along the x-axis will be used.
    Y_Length: int | None
        The length of the output image array along the y-axis. If not provided, the
        length of the input image array along the y-axis will be used.
    Z_Length: int | None
        The length of the output image array along the z-axis. If not provided, the
        length of the input image array along the z-axis will be used.
    Order: Literal[0, 1, 2, 3, 4, 5] = 1
        The order (0-5) to use for interpolation. If Segmented Data is False,
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
    for key in list(axlengths.keys()):
        if axlengths[key] == 0:
            axlengths[key] = None
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
