"""
Tests for PyDoug's payloads module.
"""

import numpy as np

from pydoug_core import payloads


def test_ShapesData() -> None:

    shapes: payloads.ShapesData = payloads.ShapesData(
        data=[np.array([[1, 2], [3, 4]])], shape_type=["line"]
    )
    assert np.all(shapes.data[0] == np.array([[1, 2], [3, 4]]))
    assert shapes.shape_type == ["line"]
