"""
Measured and calculated values of images.
"""

from typing import Annotated

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["mesh_surface_area", "label_count"]


@register_operation(
    category="measure",
    display_name="Mesh Surface Area",
    layer_type="surface",
    output_kind="value",
)
def mesh_surface_area(
    surface: tuple,
    Pixel_Size: Annotated[float, {"min": 0, "max": 1000000, "step": 0.0001}] = 1,
) -> float:  # pragma: no cover
    """
    Compute the surface area of a mesh. Convenience wrapper around scikit-image's
    measure.mesh_surface_area:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.mesh_surface_area

    Parameters
    ----------
    surface: tuple
        The surface tuple.
    Pixel_Size: float
        The edge length of the pixels in real-world units.

    Returns
    -------
    surf_area: float
        The surface area.
    """

    from skimage.measure import mesh_surface_area

    return mesh_surface_area(surface[0], surface[1]) * (Pixel_Size**2)


@register_operation(
    category="measure",
    display_name="Label Count",
    layer_type="labels",
    output_kind="value",
)
def label_count(imarray: np.ndarray) -> int:
    """
    Count the number of unique labels in a labeled segmented image. Convenience wrapper
    around numpy's max function (labels are generated sequentially, so the max value
    in a labeled image should be the number of unique labels):
    https://numpy.org/doc/stable/reference/generated/numpy.max.html

    Parameters
    ----------
    imarray: np.ndarray
        The labeled segmented image to count unique labels.

    Returns
    -------
    lab_count: int
        The unique label count.
    """

    return np.max(imarray)
