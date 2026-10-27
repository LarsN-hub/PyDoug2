"""
Tests for PyDoug's morph.features module.
"""

import numpy as np
import pytest

from pydoug_core.morph import features


def test_euclidian_dt_returns(sample_arrays) -> None:

    result1: np.ndarray = features.euclidian_dt(sample_arrays["2dgray"])
    result2: np.ndarray = features.euclidian_dt(sample_arrays["2dgray"], 2)
    result3: np.ndarray = features.euclidian_dt(sample_arrays["2drgb"])
    result4: np.ndarray = features.euclidian_dt(
        sample_arrays["2dgray"], Perform_Slicewise=True
    )
    result5: np.ndarray = features.euclidian_dt(
        sample_arrays["3dgray"], 2, Perform_Slicewise=True
    )
    assert np.all(result1 * 2 == result2)
    assert result3.shape == result1.shape
    assert np.all(result4 == result1)
    assert result5.shape == sample_arrays["3dgray"].shape


def test_mis(sample_masks, sample_2drgb_array) -> None:

    result0: np.ndarray = features.mis(sample_2drgb_array)
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

    assert np.all(result0 == sample_2drgb_array)
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

    result01: np.ndarray = features.chords(sample_masks["2dgray"])
    result02: np.ndarray = features.chords(sample_masks["3dgray"], "Z")
    assert isinstance(result01, np.ndarray)
    assert isinstance(result02, np.ndarray)
    assert result01.dtype == np.dtype(np.bool)
    assert result02.dtype == np.dtype(np.bool)
