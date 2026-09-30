"""
Manipulate the dimensions of images.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["trim_pad"]


def _refine_pad_bounds(
    imarray: np.ndarray,
    bounds_as_slices: bool,
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    z_bounds: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:

    ax_lengths: dict[str, int | None] = util.axlengths(imarray)
    bounds_dict: dict[str, tuple[int, int]] = {
        "x": x_bounds,
        "y": y_bounds,
        "z": z_bounds,
    }

    if bounds_as_slices:
        for key in ("x", "y", "z"):
            if ax_lengths[key] is None:
                continue
            if bounds_dict[key][0] < 0:
                bounds_dict[key] = (0, bounds_dict[key][1])
            if bounds_dict[key][1] < 0:
                bounds_dict[key] = (bounds_dict[key][0], 0)

    else:
        for key in ("x", "y", "z"):
            if ax_lengths[key] is None:
                continue
            if bounds_dict[key][1] < ax_lengths[key]:  # type: ignore
                bounds_dict[key] = (bounds_dict[key][0], ax_lengths[key])  # type: ignore
            combined_value: int = bounds_dict[key][1] - ax_lengths[key]  # type: ignore
            bounds_dict[key] = (
                round(combined_value / 2),
                combined_value - round(combined_value / 2),
            )

    return bounds_dict["x"], bounds_dict["y"], bounds_dict["z"]


def _refine_trim_bounds(
    imarray: np.ndarray,
    bounds_as_slices: bool,
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    z_bounds: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:

    ax_lengths: dict[str, int | None] = util.axlengths(imarray)
    bounds_dict: dict[str, tuple[int, int]] = {
        "x": x_bounds,
        "y": y_bounds,
        "z": z_bounds,
    }

    if bounds_as_slices:
        for key in ("x", "y", "z"):
            if ax_lengths[key] is None:
                continue
            if bounds_dict[key][0] < 0:
                bounds_dict[key] = (0, bounds_dict[key][1])
            if bounds_dict[key][0] >= ax_lengths[key]:  # type: ignore
                bounds_dict[key] = (ax_lengths[key] - 1, bounds_dict[key][1])  # type: ignore
            if bounds_dict[key][1] < 0:
                bounds_dict[key] = (bounds_dict[key][0], 0)
            if bounds_dict[key][1] >= ax_lengths[key]:  # type: ignore
                bounds_dict[key] = (bounds_dict[key][0], ax_lengths[key] - 1)  # type: ignore
            if sum(bounds_dict[key]) > ax_lengths[key] - 1:  # type: ignore
                while True:
                    bounds_dict[key] = (
                        max(bounds_dict[key][0] - 1, 0),
                        max(bounds_dict[key][1] - 1, 0),
                    )
                    if sum(bounds_dict[key]) <= ax_lengths[key] - 1:  # type: ignore
                        break
            bounds_dict[key] = (
                bounds_dict[key][0],
                ax_lengths[key] - bounds_dict[key][1],  # type: ignore
            )

    else:
        for key in ("x", "y", "z"):
            if ax_lengths[key] is None:
                continue
            if bounds_dict[key][1] <= 0:
                bounds_dict[key] = (bounds_dict[key][0], 1)
            if bounds_dict[key][1] > ax_lengths[key]:  # type: ignore
                bounds_dict[key] = (bounds_dict[key][0], ax_lengths[key])  # type: ignore
            left_val: int = round((ax_lengths[key] - bounds_dict[key][1]) / 2)  # type: ignore
            bounds_dict[key] = (left_val, ax_lengths[key] - left_val)  # type: ignore

    return bounds_dict["x"], bounds_dict["y"], bounds_dict["z"]


@register_operation(
    category="manipulate",
    display_name="Trim / Pad",
    allow_in_place=True,
)
def trim_pad(
    imarray: np.ndarray,
    Method: Literal["Pad", "Trim"] = "Trim",
    Bounds_as_Slices: bool = False,
    X_Bounds: tuple[
        Annotated[int, {"min": 0, "max": 100000}],
        Annotated[int, {"min": 0, "max": 100000}],
    ] = (0, 0),
    Y_Bounds: tuple[
        Annotated[int, {"min": 0, "max": 100000}],
        Annotated[int, {"min": 0, "max": 100000}],
    ] = (0, 0),
    Z_Bounds: tuple[
        Annotated[int, {"min": 0, "max": 100000}],
        Annotated[int, {"min": 0, "max": 100000}],
    ] = (0, 0),
    Padded_Color: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    In_Place: bool = False,
) -> np.ndarray:
    """
    Trim or pad the dimensions of an image. Convenience wrapper around numpy's pad,
    squeeze, and slice indexing:
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy-pad
    https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html#numpy-squeeze
    https://numpy.org/doc/stable/user/basics.indexing.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to be trimmed or padded.
    Method: Literal["Pad", "Trim"]
        The method to use (padding or trimming). Default is "Trim".
    Bounds_as_Slices: bool
        Whether the input bounds should be treated as slices or dimensions.
        True will trim/pad the input number of slices from/to each axis.
        False will trim/pad the input image to the specified dimensions on each axis.
        Default is False.
    X_Bounds: tuple
        The bounds to use for the trimming/padding the x-axis.
        If "Bounds as Slices" is True, the left bound will be trimmed/padded from/to the
        left side of the x-axis and the right bound will be trimmed/padded from/to the
        right side of the x-axis.
        If "Bounds as Slices" is False, only the right bound will be used as the final
        dimension length of the x-axis. Values are trimmed/padded evenly from each side.
    Y_Bounds: tuple
        The bounds to use for the trimming/padding the y-axis.
        If "Bounds as Slices" is True, the left bound will be trimmed/padded from/to the
        left side of the y-axis and the right bound will be trimmed/padded from/to the
        right side of the y-axis.
        If "Bounds as Slices" is False, only the right bound will be used as the final
        dimension length of the y-axis. Values are trimmed/padded evenly from each side.
    Z_Bounds: tuple
        The bounds to use for the trimming/padding the z-axis.
        If "Bounds as Slices" is True, the left bound will be trimmed/padded from/to the
        left side of the z-axis and the right bound will be trimmed/padded from/to the
        right side of the z-axis.
        If "Bounds as Slices" is False, only the right bound will be used as the final
        dimension length of the z-axis. Values are trimmed/padded evenly from each side.
        Ignored if the image is not 3D.
    Padded_Color: int
        What intensity to use for added pixels if "Method" is "Pad". Default is 0.
    In_Place: bool
        Whether or not to perform the operation on the input image (faster, saves RAM,
        but is non-reversible) or a copy (slower, consumes RAM, but is reversible).
        Default is False.

    Returns
    -------
    tparray: np.ndarray
        The trimmed or padded image.
    """

    if X_Bounds == (0, 0) and Y_Bounds == (0, 0):
        if util.is3d(imarray) and Z_Bounds == (0, 0):
            return imarray
        elif not util.is3d(imarray):
            return imarray

    if Method == "Pad":
        X_Bounds, Y_Bounds, Z_Bounds = _refine_pad_bounds(
            imarray, Bounds_as_Slices, X_Bounds, Y_Bounds, Z_Bounds
        )
        if util.is3d(imarray):
            bounds: tuple = (Z_Bounds, Y_Bounds, X_Bounds)
        else:
            bounds: tuple = (Y_Bounds, X_Bounds)
        if In_Place:
            imarray = np.pad(imarray, bounds, constant_values=Padded_Color)
            return imarray
        else:
            tparray = np.pad(imarray, bounds, constant_values=Padded_Color)
            return tparray

    else:
        X_Bounds, Y_Bounds, Z_Bounds = _refine_trim_bounds(
            imarray, Bounds_as_Slices, X_Bounds, Y_Bounds, Z_Bounds
        )
        if util.is3d(imarray):
            return np.squeeze(
                imarray[
                    Z_Bounds[0] : Z_Bounds[1],
                    Y_Bounds[0] : Y_Bounds[1],
                    X_Bounds[0] : X_Bounds[1],
                ]
            )
        else:
            return np.squeeze(
                imarray[
                    Y_Bounds[0] : Y_Bounds[1],
                    X_Bounds[0] : X_Bounds[1],
                ]
            )


@register_operation(category="manipulate", display_name="Make 3D")
def make_3d(
    imarray: np.ndarray, Slice_Count: Annotated[int, {"min": 2}] = 10
) -> np.ndarray:
    """
    Extend a 2D image along the z-axis to a specified slice count. If the image is 3D
    already then this returns the image as is. Convenience wrapper around numpy's
    repeat and expand_dims:
    https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy-repeat
    https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy-expand-dims

    Parameters
    ----------
    imarray: np.ndarray
        The 2D image to be extended.
    Slice_Count: int
        The number of slices the 3D image should have.

    Returns
    -------
    im3d: np.ndarray
        The 3D image.
    """

    if util.is3d(imarray):
        return imarray

    return np.repeat(np.expand_dims(imarray, 0), Slice_Count, axis=0)


@register_operation(
    category="manipulate",
    display_name="Append",
    array_params=("Append_Layer",),
)
def join(imarray: np.ndarray, Append_Layer: np.ndarray) -> np.ndarray:
    """
    Append an image to the back of an image (slicewise). Images may be 2D or 3D but must
    have the same xy dimensions. Only appends along the z-axis.
    Convenience wrapper around numpy's append and expand_dims:
    https://numpy.org/doc/stable/reference/generated/numpy.append.html#numpy-append
    https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy-expand-dims

    Parameters
    ----------
    imarray: np.ndarray
        The image that will be the front-most slices of the appended 3D stack.
    Append_Layer: np.ndarray
        The image array (2D or 3D) that will be appended behind the first image array.

    Returns
    -------
    append_array: np.ndarray
        The joined images.
    """

    if not util.is3d(imarray) and util.is3d(Append_Layer):
        return np.append(np.expand_dims(imarray, 0), Append_Layer, axis=0)
    elif util.is3d(imarray) and not util.is3d(Append_Layer):
        return np.append(imarray, np.expand_dims(Append_Layer, 0), axis=0)
    elif not util.is3d(imarray) and not util.is3d(Append_Layer):
        return np.append(
            np.expand_dims(imarray, axis=0), np.expand_dims(Append_Layer, 0), axis=0
        )
    return np.append(imarray, Append_Layer, axis=0)
