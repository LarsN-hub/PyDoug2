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

    Returns
    -------
    cmap: dict | DirectLabelColormap
        The napari-readable colormap.
    """

    if np.issubdtype(imarray.dtype, np.floating):
        raise ValueError("Image has floating dtype, colormap not created!")

    min_val: int = int(np.min(imarray[imarray != 0]))
    max_val: int = int(np.max(imarray))
    if c_lims[0] == c_lims[1] == 0:
        gray_count: int = max_val - min_val + 1
        keys: np.ndarray = np.arange(min_val, max_val + 1)
        carray: np.ndarray = colormaps[color_map](np.linspace(0, 1, gray_count))
    else:
        gray_count: int = c_lims[1] - c_lims[0] + 1
        if gray_count <= 0:
            raise ValueError("Upper color limit must be >= Lower limit + 1!")
        keys: np.ndarray = np.arange(
            min(min_val, c_lims[0]), max(max_val, c_lims[1]) + 1
        )
        carray = colormaps[color_map](np.linspace(0, 1, gray_count))
        if c_lims[0] - min_val >= 0:
            carray = np.append(
                np.repeat(carray[0:1], (c_lims[0] - min_val), axis=0), carray, axis=0
            )
        if max_val - c_lims[1] >= 0:
            carray = np.append(
                carray, np.repeat(carray[-1:], (max_val - c_lims[1]), axis=0), axis=0
            )

    cdict: dict[int | None, np.ndarray] = {k: v for k, v in zip(keys, carray)}
    cdict[None] = np.array([0, 0, 0, 0])
    cdict[0] = np.array([0, 0, 0, 0])

    return cdict


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
        Ignored if both are 0 and uses the max and min of the image's values
        (disregarding 0).

    Returns
    -------
    vis_result: VisualResult
        The VisualResult object.
    """

    return VisualResult(
        imarray,
        "labels",
        {
            "colormap": get_cmap(imarray, Color_Map, C_Lims),
            "iso_gradient_mode": "smooth",
            "metadata": {"custom colormap": True},
            "opacity": 1,
            "rendering": "iso_categorical",
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

    Returns
    -------
    vis_result: VisualResult
        The VisualResult object.
    """

    if Axis == "Z" and not is3d(imarray):
        raise ValueError("Specified 'Z' axis but image is not 3D!")

    mult_array = np.arange(0, imarray.shape[axstr2int(imarray, Axis)]) + 1
    mult_array = mult_array.astype(min_int_dtype(mult_array))
    expanded_axes: list[int] = []
    eval_axes: tuple = (0, 1, 2) if is3d(imarray) else (0, 1)
    for ax in eval_axes:
        if ax != axstr2int(imarray, Axis):
            expanded_axes.append(ax)
    mult_array = np.expand_dims(mult_array, tuple(expanded_axes))
    grad_array: np.ndarray = (mult_array * (imarray != 0)).astype(mult_array.dtype)

    return VisualResult(
        grad_array,
        "labels",
        {
            "colormap": get_cmap(grad_array, Color_Map, C_Lims),
            "iso_gradient_mode": "smooth",
            "metadata": {"custom colormap": True},
            "opacity": 1,
            "rendering": "iso_categorical",
        },
    )
