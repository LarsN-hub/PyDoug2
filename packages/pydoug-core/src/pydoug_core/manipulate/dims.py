"""
Manipulate the dimensions of images.
"""

from math import ceil, floor
from typing import Annotated, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["refine_pad_bounds", "refine_trim_bounds", "trim_pad", "make_3d", "join"]


def refine_pad_bounds(
    imarray: np.ndarray,
    bounds_as_slice_indices: bool,
    x_bounds: list[int],
    y_bounds: list[int],
    z_bounds: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """
    Refine user input pad bounds into trim_pad readable pad bounds.

    Parameters
    ----------
    imarray: np.ndarray
        The image that will be padded.
    bounds_as_slice_indices: bool
        True: Treat bounds as [ignore, target_ax_length]. Convert to False format below.
        False: Treat bounds as [front_pad, end_pad]. Return as such.
    x_bounds: list[int]
        List of user input x-axis bounds.
        If "bounds_as_slice_indices" is "True", format is [ignore, target_x_length].
        If "bounds_as_slice_indices" is "False", format is [front_pad, end_pad].
    y_bounds: list[int]
        List of user input y-axis bounds.
        If "bounds_as_slice_indices" is "True", format is [ignore, target_y_length].
        If "bounds_as_slice_indices" is "False", format is [front_pad, end_pad].
    z_bounds: list[int]
        List of user input z-axis bounds.
        If "bounds_as_slice_indices" is "True", format is [ignore, target_z_length].
        If "bounds_as_slice_indices" is "False", format is [front_pad, end_pad].

    Returns
    -------
    bounds: tuple
        Tuple of each axis bounds as lists [front_pad, end_pad].
    """

    ax_lengths: dict[str, int] = util.axlengths(imarray)
    bounds_dict: dict[str, list[int]] = {
        "x": x_bounds,
        "y": y_bounds,
        "z": z_bounds,
    }

    for key in ("x", "y", "z"):
        if bounds_as_slice_indices:
            target_length: int = (
                ax_lengths[key]
                if bounds_dict[key][1] < ax_lengths[key]
                else bounds_dict[key][1]
            )
            if target_length > 0 and ax_lengths[key] == 0:
                bounds_dict[key][0] = floor((target_length - ax_lengths[key] - 1) / 2)
                bounds_dict[key][1] = ceil((target_length - ax_lengths[key] - 1) / 2)
            else:
                bounds_dict[key][0] = floor((target_length - ax_lengths[key]) / 2)
                bounds_dict[key][1] = ceil((target_length - ax_lengths[key]) / 2)

        else:
            bounds_dict[key][0] = 0 if bounds_dict[key][0] < 0 else bounds_dict[key][0]
            bounds_dict[key][1] = 0 if bounds_dict[key][1] < 0 else bounds_dict[key][1]

    return bounds_dict["x"], bounds_dict["y"], bounds_dict["z"]


def refine_trim_bounds(
    imarray: np.ndarray,
    bounds_as_slice_indices: bool,
    x_bounds: list[int],
    y_bounds: list[int],
    z_bounds: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """
    Refine user input trim bounds into trim_pad readable trim bounds.

    Parameters
    ----------
    imarray: np.ndarray
        The image that will be trimmed.
    bounds_as_slice_indices: bool
        True: Treat bounds as [start_index, end_index). Return as such.
        False: Treat bounds as [front_trim, end_trim]. Convert to True format above.
    x_bounds: list[int]
        List of user input x-axis bounds.
        If "bounds_as_slice_indices" is "True", format is [start_index, end_index).
        If "bounds_as_slice_indices" is "False", format is [front_trim, end_trim].
    y_bounds: list[int]
        List of user input y-axis bounds.
        If "bounds_as_slice_indices" is "True", format is [start_index, end_index).
        If "bounds_as_slice_indices" is "False", format is [front_trim, end_trim].
    z_bounds: list[int]
        List of user input z-axis bounds.
        If "bounds_as_slice_indices" is "True", format is [start_index, end_index).
        If "bounds_as_slice_indices" is "False", format is [front_trim, end_trim].

    Returns
    -------
    bounds: tuple
        Tuple of each axis bounds as lists [start_index, end_index).
    """

    ax_lengths: dict[str, int] = util.axlengths(imarray)
    bounds_dict: dict[str, list[int]] = {
        "x": x_bounds,
        "y": y_bounds,
        "z": z_bounds,
    }

    for key in ("x", "y", "z"):
        if bounds_dict[key] == [0, 0]:
            bounds_dict[key][1] = ax_lengths[key]
            continue
        if ax_lengths[key] == 0:
            bounds_dict[key] = [0, 0]
            continue

        if bounds_as_slice_indices:
            for index in range(0, len(bounds_dict[key])):
                bounds_dict[key][index] = (
                    0 if bounds_dict[key][index] < 0 else bounds_dict[key][index]
                )
                bounds_dict[key][index] = (
                    ax_lengths[key]
                    if bounds_dict[key][index] > ax_lengths[key]
                    else bounds_dict[key][index]
                )
            bounds_dict[key][1] = (
                bounds_dict[key][0]
                if bounds_dict[key][0] > bounds_dict[key][1]
                else bounds_dict[key][1]
            )
            bounds_dict[key][0] = (
                ax_lengths[key] - 1
                if bounds_dict[key][0] == ax_lengths[key]
                else bounds_dict[key][0]
            )
            bounds_dict[key][1] = (
                bounds_dict[key][0] + 1
                if bounds_dict[key][1] == bounds_dict[key][0]
                else bounds_dict[key][1]
            )

        else:
            bounds_dict[key][1] = 1 if bounds_dict[key][1] < 1 else bounds_dict[key][1]
            bounds_dict[key][1] = (
                ax_lengths[key]
                if bounds_dict[key][1] > ax_lengths[key]
                else bounds_dict[key][1]
            )
            trim_amount: int = ax_lengths[key] - bounds_dict[key][1]
            bounds_dict[key][0] = floor(trim_amount / 2)
            bounds_dict[key][1] = ax_lengths[key] - ceil(trim_amount / 2)

    return bounds_dict["x"], bounds_dict["y"], bounds_dict["z"]


@register_operation(
    category="manipulate",
    display_name="Trim / Pad",
    allow_in_place=True,
)
def trim_pad(
    imarray: np.ndarray,
    Method: Literal["Pad", "Trim"] = "Trim",
    Bounds_as_Slice_Indices: bool = True,
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
    Trim or pad the dimensions of an image. Pat attention to [] vs. () inclusivity vs.
    exclusivity in the bounds descriptions. Convenience wrapper around numpy's pad and
    slice indexing:
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    https://numpy.org/doc/stable/user/basics.indexing.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to be trimmed or padded.
    Method: Literal["Pad", "Trim"]
        The method to use (padding or trimming). Default is "Trim".
    Bounds_as_Slice_Indices: bool
        Whether the input bounds should be treated as slice indices in the current image
        (True) or slice amounts to trim/pad to/from each axis (False). Default is False.
    X_Bounds: tuple
        The x-axis bounds.
        If "Method" is "Trim":
        If "Bounds as Slice Indices" is "True": [start_index, end_index).
        If "Bounds as Slice Indices" is "False": [front_trim_amount, end_trim_amount].
        For convenience, if "Bounds as Slice Indices" is "True" and both bounds are the
        same value, only that index is returned (no need for +1 exclusive upper bound).
        This can be used to easily return a size 1 axis.
        If "Method" is "Pad":
        If "Bounds as Slice Indices" is "True": (ignored, target_axis_length].
        If "Bounds as Slice Indices" is "False": [front_pad_amount, end_pad_amount].
        Note that [] is inclusive and () is exclusive.
    Y_Bounds: tuple
        The y-axis bounds.
        If "Method" is "Trim":
        If "Bounds as Slice Indices" is "True": [start_index, end_index).
        If "Bounds as Slice Indices" is "False": [front_trim_amount, end_trim_amount].
        For convenience, if "Bounds as Slice Indices" is "True" and both bounds are the
        same value, only that index is returned (no need for +1 exclusive upper bound).
        This can be used to easily return a size 1 axis.
        If "Method" is "Pad":
        If "Bounds as Slice Indices" is "True": (ignored, target_axis_length].
        If "Bounds as Slice Indices" is "False": [front_pad_amount, end_pad_amount].
        Note that [] is inclusive and () is exclusive.
    Z_Bounds: tuple
        The z-axis bounds.
        If "Method" is "Trim":
        If "Bounds as Slice Indices" is "True": [start_index, end_index).
        If "Bounds as Slice Indices" is "False": [front_trim_amount, end_trim_amount].
        For convenience, if "Bounds as Slice Indices" is "True" and both bounds are the
        same value, only that index is returned (no need for +1 exclusive upper bound).
        This can be used to easily return a size 1 axis.
        For z-axis only, if the returned image has 1 slice, it will convert to 2D.
        If "Method" is "Pad":
        If "Bounds as Slice Indices" is "True": (ignored, target_axis_length].
        If "Bounds as Slice Indices" is "False": [front_pad_amount, end_pad_amount].
        Note that [] is inclusive and () is exclusive.
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
            if Method == "Trim":
                return imarray
            elif Method == "Pad" and Z_Bounds == (0, 0):
                return imarray

    if In_Place:
        tparray: np.ndarray = imarray
    else:
        tparray = np.copy(imarray)

    if Method == "Pad":
        x_bounds, y_bounds, z_bounds = refine_pad_bounds(
            tparray,
            Bounds_as_Slice_Indices,
            list(X_Bounds),
            list(Y_Bounds),
            list(Z_Bounds),
        )
        if util.is3d(tparray):
            bounds: list[list[int]] = [z_bounds, y_bounds, x_bounds]
        else:
            bounds = [y_bounds, x_bounds]
        if x_bounds == y_bounds == z_bounds == [0, 0]:
            return tparray
        if util.isrgb(imarray):
            bounds.append([0, 0])
        tparray = np.pad(tparray, bounds, constant_values=Padded_Color)
        if not util.is3d(tparray) and z_bounds != [0, 0]:
            bounds.insert(0, z_bounds)
            bounds[1] = [0, 0]
            bounds[2] = [0, 0]
            tparray = np.pad(
                np.expand_dims(tparray, axis=0),
                bounds,
                constant_values=Padded_Color,
            )
        return tparray

    x_bounds, y_bounds, z_bounds = refine_trim_bounds(
        tparray, Bounds_as_Slice_Indices, list(X_Bounds), list(Y_Bounds), list(Z_Bounds)
    )
    if util.is3d(tparray):
        if (
            x_bounds == [0, imarray.shape[2]]
            and y_bounds == [0, imarray.shape[1]]
            and z_bounds == [0, imarray.shape[0]]
        ):
            return tparray
        tparray = tparray[
            z_bounds[0] : z_bounds[1],
            y_bounds[0] : y_bounds[1],
            x_bounds[0] : x_bounds[1],
        ]
        if tparray.shape[0] == 1:
            tparray = tparray[0]
    else:
        if x_bounds == [0, imarray.shape[1]] and y_bounds == [0, imarray.shape[0]]:
            return tparray
        tparray = tparray[
            y_bounds[0] : y_bounds[1],
            x_bounds[0] : x_bounds[1],
        ]
    return tparray


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
