"""
Create labels layers with controlled colors.
"""

from typing import Annotated

import cmasher  # noqa: F401 — imported for its colormap expansion effect
import numpy as np
from matplotlib import colormaps

from pydoug_core.registry import register_operation
from pydoug_core.results import VisualResult


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
    Gradiate colors in an image based on the grayscale intensity value. Note that the
    image must have an integer dtype.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to gradiate colors in.
    Color_Map: str
        The color map to use.
    C_Lims: tuple[int, int]
        The intensity limits for the bounds of the color map in the form [min, max].
        Ignored if both are 0 and uses the max and min of the image's dtype.

    Returns
    -------
    vis_result: VisualResult
        The VisualResult object.
    """

    if not np.issubdtype(imarray.dtype, np.integer):
        raise ValueError("Image dtype is not integer!")

    gray_count = np.iinfo(imarray.dtype).max - np.iinfo(imarray.dtype).min
    if np.iinfo(imarray.dtype).min == 0:
        gray_count += 1
    if C_Lims[0] == C_Lims[1] == 0:
        carray: np.ndarray = colormaps[Color_Map](np.linspace(0, 1, gray_count))
    else:
        print(True)
        r_start: int = (
            (min(C_Lims) + np.iinfo(imarray.dtype).min)
            if np.iinfo(imarray.dtype).min != 0
            else min(C_Lims)
        )
        r_end: int = (
            (max(C_Lims) + np.iinfo(imarray.dtype).min)
            if np.iinfo(imarray.dtype).min != 0
            else max(C_Lims)
        )
        carray: np.ndarray = np.empty((gray_count, 4))
        carray[r_start: r_end + 1, :] = colormaps[Color_Map](
            np.linspace(0, 1, r_end - r_start + 1)
        )
        carray[0:r_start] = np.repeat(carray[r_start, :], r_start - 1, 1)
        carray[r_end:] = np.repeat(
            carray[r_end, :], np.iinfo(imarray.dtype).max - r_end + 1, 1
        )

    return VisualResult(
        imarray,
        "image",
        {"colormap":
            {
                "colors": carray,
                "name": f"{Color_Map}_custom",
                "interpolation": "linear",
            }
        },
    )
