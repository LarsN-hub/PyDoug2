"""
Convert napari layers into plain data for non-napari operations.
"""

from typing import Any

import numpy as np
from napari.layers import Image, Labels, Shapes, Surface

from pydoug_core.payloads import ShapesData

LAYER_TYPE_MAP: dict = {
    "image": Image,
    "labels": Labels,
    "shapes": Shapes,
    "surface": Surface,
}


def layer_payload(layer: Any, live_shapes: bool = False):
    """
    Convert a napari layer into the plain data its operation receives.

    Parameters
    ----------
    layer: Any
        The napari layer to extract plain data from.
    live_shapes: bool
        Whether or not a shapes layer is live.

    Returns
    -------
    data: Any
        The layer data.
    """

    if isinstance(layer, Shapes):
        if live_shapes:
            return layer  # unrecorded operation may call layer methods like "to_labels"
        return ShapesData(
            data=[np.array(v, dtype=float) for v in layer.data],
            shape_type=list(layer.shape_type),
        )

    if isinstance(layer, Surface):
        return tuple(np.asarray(a) for a in layer.data if a is not None)

    return layer.data
