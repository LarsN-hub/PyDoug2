"""
Tests for PyDoug's trans.domain module.
"""

import numpy as np

from pydoug_core.trans import domain


def test_radon(sample_arrays) -> None:

    result1: np.ndarray = domain.radon(sample_arrays["2dgray"])
    result2: np.ndarray = domain.radon(sample_arrays["3dgray"])
    result3: np.ndarray = domain.radon(sample_arrays["3drgb"])
    assert result1.shape[1] == 180
    assert result2.shape[0] == sample_arrays["3dgray"].shape[0]
    assert result2.shape[2] == 180
    assert result2.shape[0] == sample_arrays["3drgb"].shape[0]
    assert result3.shape[2] == 180


def test_invradon(sample_arrays) -> None:

    result1: np.ndarray = domain.invradon(domain.radon(sample_arrays["2dgray"]))
    result2: np.ndarray = domain.invradon(domain.radon(sample_arrays["3dgray"]))
    result3: np.ndarray = domain.invradon(domain.radon(sample_arrays["3drgb"]))
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape[:] == sample_arrays["3drgb"].shape[:-1]


def test_fft(sample_arrays) -> None:

    result1: np.ndarray = domain.fft(sample_arrays["2dgray"])
    result2: np.ndarray = domain.fft(sample_arrays["3dgray"])
    result3: np.ndarray = domain.fft(sample_arrays["3dgray"], True)
    result4: np.ndarray = domain.fft(sample_arrays["3drgb"])
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert not np.all(result2 == result3)
    assert result4.shape[:] == sample_arrays["3drgb"].shape[:-1]


def test_invfft(sample_arrays) -> None:

    result1: np.ndarray = domain.invfft(domain.fft(sample_arrays["2dgray"]))
    result2: np.ndarray = domain.invfft(domain.fft(sample_arrays["3dgray"]))
    result3: np.ndarray = domain.invfft(domain.fft(sample_arrays["3dgray"], True))
    result4: np.ndarray = domain.invfft(domain.fft(sample_arrays["3dgray"]), True)
    result5: np.ndarray = domain.invfft(domain.fft(sample_arrays["3drgb"]))
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3dgray"].shape
    assert not np.all(result2 == result3)
    assert not np.all(result3 == result4)
    assert not np.all(result2 == result4)
    assert result5.shape[:] == sample_arrays["3drgb"].shape[:-1]


def test_hough_line_returns(sample_arrays) -> None:

    result1: np.ndarray | tuple = domain.hough(sample_arrays["2dgray"])
    result2: np.ndarray | tuple = domain.hough(sample_arrays["2dgray"], fullout=True)
    result3: np.ndarray | tuple = domain.hough(sample_arrays["3dgray"])
    result4: np.ndarray | tuple = domain.hough(sample_arrays["3dgray"], fullout=True)
    result5: np.ndarray | tuple = domain.hough(sample_arrays["3drgb"])
    if isinstance(result1, np.ndarray):
        assert result1.shape != sample_arrays["2dgray"].shape
    if isinstance(result2, tuple):
        assert result2[0].shape != sample_arrays["2dgray"].shape
        if isinstance(result1, np.ndarray):
            assert result2[0].shape == result1.shape
        assert isinstance(result2[1], np.ndarray)
        assert isinstance(result2[2], np.ndarray)
    if isinstance(result3, np.ndarray):
        assert result3.shape != sample_arrays["3dgray"].shape
        assert result3.shape[0] == sample_arrays["3dgray"].shape[0]
        if isinstance(result5, np.ndarray):
            assert result3.shape == result5.shape
    if isinstance(result4, tuple):
        assert result4[0].shape != sample_arrays["3dgray"].shape
        assert result4[0].shape[0] == sample_arrays["3dgray"].shape[0]
        if isinstance(result3, np.ndarray):
            assert result4[0].shape == result3.shape
        assert isinstance(result4[1], list)
        assert isinstance(result4[2], list)
        assert isinstance(result4[1][0], np.ndarray)
        assert isinstance(result4[2][0], np.ndarray)


def test_hough_circle_returns(sample_arrays) -> None:

    result1: np.ndarray | tuple = domain.hough(sample_arrays["2dgray"], "circle")
    result2: np.ndarray | tuple = domain.hough(
        sample_arrays["2dgray"], "circle", radii=np.arange(2, 4)
    )
    result3: np.ndarray | tuple = domain.hough(sample_arrays["3dgray"], "circle")
    result4: np.ndarray | tuple = domain.hough(
        sample_arrays["3dgray"], "circle", radii=np.arange(2, 4)
    )
    result5: np.ndarray | tuple = domain.hough(sample_arrays["3drgb"], "circle")
    if isinstance(result1, np.ndarray):
        assert len(result1.shape) > len(sample_arrays["2dgray"].shape)
    if isinstance(result2, np.ndarray):
        assert result2.shape[0] == len(np.arange(2, 4))
    if isinstance(result3, np.ndarray):
        assert len(result3.shape) > len(sample_arrays["3dgray"].shape)
        if isinstance(result5, np.ndarray):
            assert result3.shape == result5.shape
    if isinstance(result4, np.ndarray):
        assert result4.shape[0] == len(np.arange(2, 4))
        assert result4.shape[-1] == sample_arrays["3dgray"].shape[0]
