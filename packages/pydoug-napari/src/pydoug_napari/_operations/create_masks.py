"""
Create masks from shapes or labels layers.
"""

from typing import Annotated

import numpy as np
from napari.layers import Shapes

from pydoug_core.registry import register_operation
from pydoug_core.util import is3d

__all__ = ["shapes_to_mask"]


@register_operation(
    category="generate",
    display_name="Shapes to Mask",
    layer_type="shapes",
    recordable=False,
    array_params=("Image_Layer",),
    array_param_flags={"Image_Layer": "Use_Image_Dimensions"},
)
def shapes_to_mask(
    shapes_layer: Shapes,
    Dimensions: tuple[
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 0}],
    ] = (100, 100, 0),
    Use_Image_Dimensions: bool = True,
    Image_Layer: np.ndarray | None = None,
    Propagate_Through_Z: bool = True,
) -> np.ndarray:
    """
    Rasterize the shapes in a shapes layer into a mask image. Convenience wrapper around
    napari's shapes layer to_labels method and numpy's any and stack:
    https://napari.org/stable/api/napari.layers.Shapes.html#napari.layers.Shapes.to_labels
    https://numpy.org/doc/stable/reference/generated/numpy.any.html
    https://numpy.org/doc/stable/reference/generated/numpy.stack.html

    Parameters
    ----------
    shapes_layer: Shapes
        The shapes layer to convert to a binary mask.
    Dimensions: tuple[int, int, int]
        The dimensions of the output mask image in the format [X, Y, Z]. Ignored if
        "Use Image Dimensions" is "True".
    Use_Image_Dimensions: bool
        Whether to use another image layer's dimensions as the mask image's dimensions
        (True) or to use the input dimensions (False). Default is True.
    Image_Layer: np.ndarray
        If "Use Image Dimensions" is "True", the image layer whose dimensions should be
        used for the output mask image. If not provided, "Dimensions" will be used even
        if "Use Image Dimensions" is "True".
    Propagate_Through_Z: bool
        Whether to propagate shapes along the z-axis if the image is 3D (True) or to
        leave each shape on the slice it was created on (False). Default is True.

    Returns
    -------
    mask: np.ndarray
        The mask image.
    """

    if Use_Image_Dimensions and Image_Layer is not None:
        dims: tuple = Image_Layer.shape
    else:
        if Dimensions[-1] == 0:
            dims = (Dimensions[1], Dimensions[0])
        else:
            dims = (Dimensions[2], Dimensions[1], Dimensions[0])

    mask: np.ndarray = shapes_layer.to_labels(labels_shape=dims) > 0

    if Propagate_Through_Z and is3d(mask):
        return np.stack([mask.any(axis=0)] * mask.shape[0], axis=0)

    return mask


@register_operation(
    category="generate",
    display_name="Labels to Mask",
    layer_type="labels",
    recordable=False,
)
def labels_to_mask(labels: np.ndarray) -> np.ndarray:
    """
    Convert a labels layer into a mask image. Useful for painting a mask in a labels
    layer and then converting to a usable image layer for other widgets.

    Parameters
    ----------
    labels: np.ndarray
        The labels image to convert to a mask. All values > 0 are set to True while 0
        values are set to False.

    Returns
    -------
    mask: np.ndarray
        The mask image.
    """

    return labels > 0
