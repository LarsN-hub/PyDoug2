"""
Convert napari layers into plain data for non-napari operations.
"""

import numpy as np
from napari.layers import Image, Labels, Shapes, Surface

LAYER_TYPE_MAP: dict = {
    "image": Image,
    "labels": Labels,
    "shapes": Shapes,
    "surface": Surface,
}


def layer_payload(layer):
    """
    Convert a napari layer into the plain data its operation receives.

    Parameters
    ----------
    layer: Any
        The napari layer to extract plain data from.

    Returns
    -------
    data: Any
        The layer data.
    """

    if isinstance(layer, Shapes):
        return layer

    if isinstance(layer, Surface):
        return tuple(np.asarray(a) for a in layer.data if a is not None)

    return layer.data
