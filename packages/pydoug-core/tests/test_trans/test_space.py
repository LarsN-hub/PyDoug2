"""
Tests for PyDoug's trans.domain module.
"""

import numpy as np

from pydoug_core.trans import space


def test_fft(sample_arrays) -> None:

    result1: np.ndarray = space.fft(sample_arrays["2dgray"])
    result2: np.ndarray = space.fft(sample_arrays["3dgray"])
    result3: np.ndarray = space.fft(sample_arrays["3dgray"], Perform_Slicewise=True)
    result4: np.ndarray = space.fft(sample_arrays["3drgb"])
    result5: np.ndarray = space.fft(result1, True)
    result6: np.ndarray = space.fft(result2, True)
    result7: np.ndarray = space.fft(result2, True, True)
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert not np.all(result2 == result3)
    assert result4.shape[:] == sample_arrays["3drgb"].shape[:-1]
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["3dgray"].shape


def test_radon(sample_arrays) -> None:

    result1: np.ndarray = space.radon(sample_arrays["2dgray"])
    result2: np.ndarray = space.radon(sample_arrays["3dgray"])
    result3: np.ndarray = space.radon(sample_arrays["3drgb"])
    result4: np.ndarray = space.radon(result1, True)
    result5: np.ndarray = space.radon(result2, True)
    result6: np.ndarray = space.radon(result3, True)
    assert result1.shape[1] == 180
    assert result2.shape[0] == sample_arrays["3dgray"].shape[0]
    assert result2.shape[2] == 180
    assert result2.shape[0] == sample_arrays["3drgb"].shape[0]
    assert result3.shape[2] == 180
    assert result4.shape == sample_arrays["2dgray"].shape
    assert result5.shape == sample_arrays["3dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape


def test_hough(sample_arrays) -> None:

    result1: np.ndarray = space.hough(sample_arrays["2dgray"])
    result2: np.ndarray = space.hough(sample_arrays["3dgray"])
    result3: np.ndarray = space.hough(sample_arrays["2drgb"])
    result4: np.ndarray = space.hough(sample_arrays["3drgb"])
    result5: np.ndarray = space.hough(sample_arrays["2dgray"], "Circle")
    result6: np.ndarray = space.hough(sample_arrays["3dgray"], "Circle")
    result7: np.ndarray = space.hough(sample_arrays["2drgb"], "Circle")
    result8: np.ndarray = space.hough(sample_arrays["3drgb"], "Circle")
    assert result1.shape == result3.shape
    assert result2.shape == result4.shape
    assert result5.shape == result7.shape
    assert result6.shape == result8.shape
