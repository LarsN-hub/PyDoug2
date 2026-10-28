"""
Tests for PyDoug's morph.features module.
"""

import numpy as np
import pytest

from pydoug_core.morph import features


def test_euclidian_dt(sample_arrays) -> None:

    result1: np.ndarray = features.euclidian_dt(sample_arrays["2dgray"])
    result2: np.ndarray = features.euclidian_dt(sample_arrays["2dgray"], 2)
    with pytest.raises(ValueError):
        _ = features.euclidian_dt(sample_arrays["2drgb"])
    result4: np.ndarray = features.euclidian_dt(
        sample_arrays["2dgray"], Perform_Slicewise=True
    )
    result5: np.ndarray = features.euclidian_dt(
        sample_arrays["3dgray"], 2, Perform_Slicewise=True
    )
    assert np.all(result1 * 2 == result2)
    assert np.all(result4 == result1)
    assert result5.shape == sample_arrays["3dgray"].shape


def test_mis(sample_masks, sample_2drgb_array) -> None:

    with pytest.raises(ValueError):
        _ = features.mis(sample_2drgb_array)
    result1: np.ndarray = features.mis(sample_masks["2dgray"])
    result2: np.ndarray = features.mis(sample_masks["3dgray"])
    result3: np.ndarray = features.mis(sample_masks["2dgray"], Perform_Slicewise=True)
    result4: np.ndarray = features.mis(sample_masks["3dgray"], Perform_Slicewise=True)
    result5: np.ndarray = features.mis(sample_masks["2dgray"], "Brute Force")
    result6: np.ndarray = features.mis(sample_masks["3dgray"], "Brute Force")
    result7: np.ndarray = features.mis(
        sample_masks["2dgray"], "Brute Force", Perform_Slicewise=True
    )
    result8: np.ndarray = features.mis(
        sample_masks["3dgray"], "Brute Force", Perform_Slicewise=True
    )
    result9: np.ndarray = features.mis(sample_masks["2dgray"], "ImageJ")
    result10: np.ndarray = features.mis(sample_masks["3dgray"], "ImageJ")
    result11: np.ndarray = features.mis(
        sample_masks["2dgray"], "ImageJ", Perform_Slicewise=True
    )
    result12: np.ndarray = features.mis(
        sample_masks["3dgray"], "ImageJ", Perform_Slicewise=True
    )
    result13: np.ndarray = features.mis(sample_masks["2dgray"], "Convolution")
    result14: np.ndarray = features.mis(sample_masks["3dgray"], "Convolution")
    result15: np.ndarray = features.mis(
        sample_masks["2dgray"], "Convolution", Perform_Slicewise=True
    )
    result16: np.ndarray = features.mis(
        sample_masks["3dgray"], "Convolution", Perform_Slicewise=True
    )
    result17: np.ndarray = features.mis(sample_masks["2dgray"], Return_Diameters=True)

    assert np.all(result1 == result3)
    assert np.all(result5 == result7)
    assert np.all(result9 == result11)
    assert np.all(result13 == result15)
    assert np.all(result1 * 2 == result17)
    assert result1.dtype == np.float64
    assert result2.dtype == np.float64
    assert result3.dtype == np.float64
    assert result4.dtype == np.float64
    assert result5.dtype == np.float64
    assert result6.dtype == np.float64
    assert result7.dtype == np.float64
    assert result8.dtype == np.float64
    assert result9.dtype == np.float64
    assert result10.dtype == np.float64
    assert result11.dtype == np.float64
    assert result12.dtype == np.float64
    assert result13.dtype == np.float64
    assert result14.dtype == np.float64
    assert result15.dtype == np.float64
    assert result16.dtype == np.float64
    assert result17.dtype == np.float64


def test_add_chords(sample_masks) -> None:

    with pytest.raises(ValueError):
        _ = features.chords(sample_masks["2dgray"], "Z")
    with pytest.raises(ValueError):
        _ = features.chords(sample_masks["2drgb"])
    result01: np.ndarray = features.chords(sample_masks["2dgray"])
    result02: np.ndarray = features.chords(sample_masks["3dgray"], "Z")
    assert isinstance(result01, np.ndarray)
    assert isinstance(result02, np.ndarray)
    assert result01.dtype == np.dtype(np.bool)
    assert result02.dtype == np.dtype(np.bool)


def test_path_metrics(sample_masks) -> None:

    test_array_2d: np.ndarray = np.array(
        [
            [1, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0],
        ]
    )
    test_array_3d: np.ndarray = np.stack([test_array_2d] * 2, axis=0)
    with pytest.raises(ValueError):
        _ = features.path_metrics(sample_masks["2drgb"], "Length")
    with pytest.raises(ValueError):
        _ = features.path_metrics(test_array_2d, "Length", "Z")
    result00: np.ndarray = features.path_metrics(test_array_2d, "Length", "Y", False, 3)
    result01: np.ndarray = features.path_metrics(test_array_2d, "Length", "X", False, 1)
    result02: np.ndarray = features.path_metrics(test_array_3d, "Length", "Z", False)
    result03: np.ndarray = features.path_metrics(
        test_array_2d, "Length", "Y", True, 1, 1
    )
    result04: np.ndarray = features.path_metrics(
        test_array_2d, "Length", "X", True, 2, 1
    )
    result05: np.ndarray = features.path_metrics(
        test_array_3d, "Length", "Z", True, 3, 1
    )
    result06: np.ndarray = features.path_metrics(
        test_array_3d, "Length", "Y", True, 3, 1
    )
    result07: np.ndarray = features.path_metrics(
        test_array_3d, "Length", "X", True, 3, 1
    )
    result08: np.ndarray = features.path_metrics(test_array_3d, "Length", "X", False)

    assert isinstance(result00, np.ndarray)
    assert isinstance(result01, np.ndarray)
    assert isinstance(result02, np.ndarray)
    assert isinstance(result03, np.ndarray)
    assert isinstance(result04, np.ndarray)
    assert isinstance(result05, np.ndarray)
    assert isinstance(result06, np.ndarray)
    assert isinstance(result07, np.ndarray)
    assert isinstance(result08, np.ndarray)
    assert result00.dtype == np.float64
    assert result01.dtype == np.float64
    assert result02.dtype == np.float64
    assert result03.dtype == np.float64
    assert result04.dtype == np.float64
    assert result05.dtype == np.float64
    assert result06.dtype == np.float64
    assert result07.dtype == np.float64
    assert result08.dtype == np.float64

    result09: np.ndarray = features.path_metrics(test_array_2d, "Density")
    result10: np.ndarray = features.path_metrics(test_array_2d, "Density", "Y", True)
    result11: np.ndarray = features.path_metrics(test_array_2d, "Density", "X", True)
    result12: np.ndarray = features.path_metrics(test_array_3d, "Density")
    result13: np.ndarray = features.path_metrics(test_array_3d, "Density", "X", True)
    result14: np.ndarray = features.path_metrics(test_array_3d, "Density", "Z", True)
    result15: np.ndarray = features.path_metrics(
        test_array_2d, "Density", Normalize_Density_Values=True
    )

    assert isinstance(result09, np.ndarray)
    assert isinstance(result10, np.ndarray)
    assert isinstance(result11, np.ndarray)
    assert isinstance(result12, np.ndarray)
    assert isinstance(result13, np.ndarray)
    assert isinstance(result14, np.ndarray)
    assert isinstance(result15, np.ndarray)
    assert result09.dtype == np.uint8
    assert result10.dtype == np.uint8
    assert result11.dtype == np.uint8
    assert result12.dtype == np.uint8
    assert result13.dtype == np.uint8
    assert result14.dtype == np.uint8
    assert result15.dtype == np.float64
