"""
Tests for PyDoug's features.shapes module.
"""

import numpy as np
from skimage.color import rgb2gray
from skimage.data import hubble_deep_field as hubble

from pydoug_core.features import shapes


def test_blobs(sample_2drgb_array) -> None:

    test_array: np.ndarray = rgb2gray(hubble()[50:100, 580:630])
    test_array3d: np.ndarray = np.stack([test_array] * 5, axis=0)
    result0: np.ndarray = shapes.blobs(sample_2drgb_array)
    result1: np.ndarray = shapes.blobs(test_array, Absolute_Threshold=0)
    result2: np.ndarray = shapes.blobs(test_array3d)
    result3: np.ndarray = shapes.blobs(test_array, "DoG")
    result4: np.ndarray = shapes.blobs(test_array3d, "DoG")
    result5: np.ndarray = shapes.blobs(test_array, "LoG")
    result6: np.ndarray = shapes.blobs(test_array3d, "LoG")
    result7: np.ndarray = shapes.blobs(
        test_array, Populate_with_Sizes=True, Return_Diameters=True
    )
    result8: np.ndarray = shapes.blobs(test_array3d, Populate_with_Sizes=True)
    result9: np.ndarray = shapes.blobs(
        test_array, "DoG", Populate_with_Sizes=True, Return_Diameters=True
    )
    result10: np.ndarray = shapes.blobs(test_array3d, "DoG", Populate_with_Sizes=True)
    result11: np.ndarray = shapes.blobs(
        test_array, "LoG", Populate_with_Sizes=True, Return_Diameters=True
    )
    result12: np.ndarray = shapes.blobs(test_array3d, "LoG", Populate_with_Sizes=True)
    result13: np.ndarray = shapes.blobs(test_array, Relative_Threshold=0.2)

    assert np.all(result0 == sample_2drgb_array)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool
    assert result5.dtype == np.bool
    assert result6.dtype == np.bool
    assert result7.dtype == np.float64
    assert result8.dtype == np.float64
    assert result9.dtype == np.float64
    assert result10.dtype == np.float64
    assert result11.dtype == np.float64
    assert result12.dtype == np.float64
    assert result13.dtype == np.bool
