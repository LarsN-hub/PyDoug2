"""
Tests for PyDoug's detect.shapes module.
"""

import numpy as np
from skimage.color import rgb2gray
from skimage.data import hubble_deep_field as hubble
from skimage.draw import circle_perimeter, line

from pydoug_core.detect import shapes


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


def test_lines() -> None:

    test_array: np.ndarray = np.zeros((25, 30), np.uint8)
    test_array[line(2, 5, 20, 15)] = 255
    test_array[line(8, 3, 13, 24)] = 255
    test_array[line(20, 4, 5, 26)] = 255
    test_array2 = np.fliplr(test_array)
    test_array3 = np.flipud(test_array)
    test_array3d = np.stack((test_array, test_array2, test_array3), axis=0)
    test_array_rgb = np.stack([test_array] * 3, axis=2)

    result0: np.ndarray = shapes.lines(test_array_rgb)
    result1: np.ndarray = shapes.lines(test_array)
    result2: np.ndarray = shapes.lines(test_array3d)
    result3: np.ndarray = shapes.lines(test_array, Probabilistic=True)
    result4: np.ndarray = shapes.lines(test_array3d, Probabilistic=True)
    result3: np.ndarray = shapes.lines(test_array, Threshold=2, Probabilistic=True)
    assert np.all(result0 == test_array_rgb)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool


def test_circles() -> None:

    test_array: np.ndarray = np.zeros((25, 30), np.uint8)
    test_array[circle_perimeter(10, 12, 4, shape=test_array.shape)] = 255
    test_array[circle_perimeter(20, 25, 2, shape=test_array.shape)] = 255
    test_array[circle_perimeter(15, 15, 10, shape=test_array.shape)] = 255
    test_array2 = np.fliplr(test_array)
    test_array3 = np.flipud(test_array)
    test_array3d = np.stack((test_array, test_array2, test_array3), axis=0)
    test_array_rgb = np.stack([test_array] * 3, axis=2)

    result0: np.ndarray = shapes.circles(test_array_rgb)
    result1: np.ndarray = shapes.circles(test_array, (2, 15, 1), 3)
    result2: np.ndarray = shapes.circles(test_array3d, (2, 15, 1), 3)
    result3: np.ndarray = shapes.circles(test_array, (2, 15, 1), 3, 2)
    assert np.all(result0 == test_array_rgb)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result1.shape == test_array.shape
    assert result2.shape == test_array3d.shape
    assert result3.shape == test_array.shape


def test_centroid(sample_arrays) -> None:

    result1: np.ndarray = shapes.centroid(sample_arrays["2dgray"])
    result2: np.ndarray = shapes.centroid(sample_arrays["3dgray"])
    result3: np.ndarray = shapes.centroid(sample_arrays["2drgb"])
    result4: np.ndarray = shapes.centroid(sample_arrays["3drgb"])

    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2dgray"].shape
    assert result4.shape == sample_arrays["3dgray"].shape
    assert np.count_nonzero(result1) == 1
    assert np.count_nonzero(result2) == 1
    assert np.count_nonzero(result3) == 1
    assert np.count_nonzero(result4) == 1
