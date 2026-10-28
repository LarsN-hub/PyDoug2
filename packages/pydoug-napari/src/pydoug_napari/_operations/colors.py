"""
Create labels layers with controlled colors.
"""

from typing import Annotated, Literal

import cmasher  # noqa: F401 — imported for its colormap expansion effect
import numpy as np
from matplotlib import colormaps

from pydoug_core.registry import register_operation
from pydoug_core.results import VisualResult
from pydoug_core.util import axstr2int, is3d, min_int_dtype

__all__ = ["get_cmap", "value_gradient", "axial_gradient"]


def get_cmap(
    imarray: np.ndarray,
    color_map: str = "inferno",
    c_lims: tuple[int, int] = (0, 0),
    return_type: Literal["Image", "Labels"] = "Image",
) -> dict:
    """
    Return a napari-readable colormap from matplotlib and cmasher colormaps for the
    input integer image, colormap string, and color limits. WARNING, this can take an
    exceptionally long time if the dtype is greater than (u)int16 (e.g., int32).
    Convenience wrapper around matplotlib's and cmasher's colormaps and numpy's
    linspace:
    https://matplotlib.org/stable/users/explain/colors/colormaps.html
    https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
    https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to create the colormap for.
    color_map: str
        The colormap string.
    c_lims: tuple
        The intensity/label limits for the color map beyond which values are clipped.
    return_type: Literal
        Whether to return an image layer colormap ("Image) or a labels layer colormap
        ("Labels). Default is "Image".

    Returns
    -------
    cmap: dict | DirectLabelColormap
        The napari-readable colormap.
    """

    if np.issubdtype(imarray.dtype, np.floating):
        raise ValueError("Image has floating dtype, value gradient not assigned!")

    min_val: int = np.iinfo(imarray.dtype).min if imarray.dtype != np.bool else 0
    max_val: int = np.iinfo(imarray.dtype).max if imarray.dtype != np.bool else 1
    gray_count: int = max_val - min_val + 1
    if c_lims[0] == c_lims[1] == 0:
        carray: np.ndarray = colormaps[color_map](np.linspace(0, 1, gray_count))
    else:
        r_start: int = (
            (round(min(c_lims)) + abs(min_val)) if min_val != 0 else round(min(c_lims))
        )
        r_end: int = (
            (round(max(c_lims)) + abs(min_val)) if min_val != 0 else round(max(c_lims))
        )
        carray: np.ndarray = np.empty((gray_count, 4))
        carray[r_start : (r_end + 1), :] = colormaps[color_map](
            np.linspace(0, 1, r_end - r_start + 1)
        )
        carray[0:r_start, :] = np.repeat(carray[r_start : (r_start + 1), :], r_start, 0)
        carray[r_end:, :] = np.repeat(
            carray[r_end : (r_end + 1), :],
            gray_count - r_end,
            0,
        )

    if return_type == "Labels":
        keys: np.ndarray = np.round(
            np.linspace(0, np.max(imarray), np.max(imarray))
        ).astype(np.int32)
        cdict: dict[int | None, np.ndarray] = {k: v for k, v in zip(keys, carray)}
        cdict[None] = np.array([0, 0, 0, 0])

        return cdict

    return {"colors": carray, "name": f"{color_map}_custom"}


@register_operation(
    category="visualize",
    display_name="Value Gradient",
    output_kind="visual",
    widget_options={"Color_Map": {"choices": list(colormaps)}},
)
def value_gradient(
    imarray: np.ndarray,
    Color_Map: str = "inferno",
    C_Lims: tuple[
        Annotated[int, {"min": -32768, "max": 65535}],
        Annotated[int, {"min": -32768, "max": 65535}],
    ] = (0, 0),
    Return_Type: Literal["Image", "Labels"] = "Image",
) -> VisualResult:
    """
    Gradiate colors in an image based on the grayscale intensity value. Image must have
    an integer dtype. Convenience wrapper around matplotlib's and cmasher's colormaps
    and numpy's linspace:
    https://matplotlib.org/stable/users/explain/colors/colormaps.html
    https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
    https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to gradiate colors in.
    Color_Map: str
        The color map to use.
    C_Lims: tuple[int, int]
        The intensity limits for the bounds of the color map in the form [min, max].
        Ignored if both are 0 and uses the max and min of the image's dtype.
    Return_Type: Literal
        Whether to return an image layer or labels layer. Default is "Image".

    Returns
    -------
    vis_result: VisualResult
        The VisualResult object.
    """

    return VisualResult(
        imarray,
        Return_Type.lower(),  # type: ignore
        {
            "colormap": get_cmap(imarray, Color_Map, C_Lims, Return_Type),
            "metadata": {"custom colormap": True},
            "opacity": 1,
        },
    )


@register_operation(
    category="visualize",
    display_name="Axial Gradient",
    output_kind="visual",
    widget_options={"Color_Map": {"choices": list(colormaps)}},
)
def axial_gradient(
    imarray: np.ndarray,
    Axis: Literal["X", "Y", "Z"] = "Y",
    Color_Map: str = "inferno",
    C_Lims: tuple[
        Annotated[int, {"min": -32768, "max": 65535}],
        Annotated[int, {"min": -32768, "max": 65535}],
    ] = (0, 0),
    Return_Type: Literal["Image", "Labels"] = "Image",
) -> VisualResult:
    """
    Gradiate colors in a segmented image based on the axial position of each segmented
    pixel. All pixels not equal to 0 are considered segmented pixels. Convenience
    wrapper around matplotlib's and cmasher's colormaps and numpy's arange, expand_dims,
    and linspace:
    https://matplotlib.org/stable/users/explain/colors/colormaps.html
    https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
    https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to gradiate colors in.
    Axis: Literal
        The axis to gradiate colors along.
    Color_Map: str
        The color map to use.
    C_Lims: tuple[int, int]
        The intensity limits for the bounds of the color map in the form [min, max].
        Ignored if both are 0 and uses the bounds of the specified axis.
    Return_Type: Literal
        Whether to return an image layer or labels layer. Default is "Image".

    Returns
    -------
    vis_result: VisualResult
        The VisualResult object.
    """

    if Axis == "Z" and not is3d(imarray):
        raise ValueError("Specified 'Z' axis but image is not 3D!")

    mult_array: np.ndarray = np.arange(1, imarray.shape[axstr2int(imarray, Axis)] + 1)
    if C_Lims[0] != 0 and C_Lims[1] != 0:
        C_Lims = (0, C_Lims[1]) if C_Lims[0] < 0 else C_Lims
        C_Lims = (
            (C_Lims[0], imarray.shape[axstr2int(imarray, Axis)])
            if C_Lims[1] > imarray.shape[axstr2int(imarray, Axis)]
            else C_Lims
        )
        mult_array: np.ndarray = np.ones(imarray.shape[axstr2int(imarray, Axis)])
        mult_array[min(C_Lims) : max(C_Lims) + 1] = np.arange(
            1, max(C_Lims) - min(C_Lims) + 2
        )
        mult_array[max(C_Lims) + 1 :] = mult_array[max(C_Lims)]
    else:
        mult_array = np.arange(1, imarray.shape[axstr2int(imarray, Axis)] + 1)
    expanded_axes: list[int] = []
    eval_axes: tuple = (0, 1, 2) if is3d(imarray) else (0, 1)
    for ax in eval_axes:
        if ax != axstr2int(imarray, Axis):
            expanded_axes.append(ax)
    mult_array = np.expand_dims(mult_array, tuple(expanded_axes))
    grad_array: np.ndarray = (mult_array * (imarray != 0)).astype(
        min_int_dtype(mult_array * (imarray != 0))
    )

    return VisualResult(
        grad_array,
        Return_Type.lower(),  # type: ignore
        {
            "colormap": get_cmap(
                grad_array, Color_Map, (0, np.max(grad_array)), Return_Type
            ),
            "metadata": {"custom colormap": True},
            "opacity": 1,
        },
    )
