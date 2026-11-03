"""
Tests for PyDoug's napari plugin _operations.convert_layers module.
"""

import numpy as np
import pytest

from pydoug_napari._operations import convert_layers


def test_convert_layers(sample_arrays) -> None:

    with pytest.raises(ValueError):
        _ = convert_layers.im2lab(sample_arrays["2dgray"].astype(np.float64))
    result01: np.ndarray = convert_layers.im2lab(sample_arrays["2dgray"])
    result02: np.ndarray = convert_layers.im2lab(sample_arrays["2drgb"])
    result03: np.ndarray = convert_layers.lab2im(sample_arrays["2dgray"])
    assert isinstance(result01, np.ndarray)
    assert isinstance(result02, np.ndarray)
    assert isinstance(result03, np.ndarray)
