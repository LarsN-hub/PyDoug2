"""
Classes to transfer napari data to core package without needing napari.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ShapesData:
    """
    The vertices and shape types of a shapes layer, without any napari objects.

    Attributes
    ----------
    data: list[np.ndarray]
        One (N, D) vertex array per shape.
    shape_type: list[str]
        The shape type ("line", "rectangle", "ellipse", "polygon", "path")
    """

    data: list[np.ndarray]
    shape_type: list[str]
