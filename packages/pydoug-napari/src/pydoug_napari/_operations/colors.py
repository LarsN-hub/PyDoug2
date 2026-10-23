"""
Create labels layers with controlled colors.
"""

from typing import Annotated, Literal

import cmasher  # noqa: F401 — imported for its colormap expansion effect
import numpy as np
from matplotlib import colormaps

from pydoug_core.registry import register_operation
from pydoug_core.results import VisualResult

__all__ = ["value_gradient"]


def get_cmap(
    imarray: np.ndarray,
    color_map: str = "inferno",
    c_lims: tuple[int, int] = (0, 0),
    return_type: Literal["Image", "Labels"] = "Image",
) -> dict:
    """
    Return a napari-readable colormap from matplotlib and cmasher colormaps for the
    input integer image, colormap string, and color limits.

    Parameters
    ----------
    imarray: np.ndarray
        The image to create the colormap for.
    color_map: str
        The colormap string.
    c_lims: tuple
        The intensity/label limits for the color map beyond which values are clipped.
    return_type: Literal
        Whether to return an image layer colormap (dict) or a labels layer colormap
        (dlc). Default is "dict" for image layers.

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
    an integer dtype.

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
