"""
Tests for PyDoug's filter.pass module.
"""

import numpy as np

from pydoug_core.filter import pass_band


def test_butter(sample_arrays) -> None:

    result1: np.ndarray = pass_band.butter(sample_arrays["2dgray"])
    result2: np.ndarray = pass_band.butter(sample_arrays["3dgray"])
    result3: np.ndarray = pass_band.butter(sample_arrays["2drgb"])
    result4: np.ndarray = pass_band.butter(sample_arrays["3drgb"])
    result5: np.ndarray = pass_band.butter(
        sample_arrays["3dgray"], Perform_Slicewise=True
    )
    result6: np.ndarray = pass_band.butter(
        sample_arrays["3drgb"], Perform_Slicewise=True
    )
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape
    assert result6.shape == sample_arrays["3drgb"].shape
    assert result1.dtype != sample_arrays["2dgray"].dtype
    assert result2.dtype != sample_arrays["3dgray"].dtype
    assert result3.dtype != sample_arrays["2drgb"].dtype
    assert result4.dtype != sample_arrays["3drgb"].dtype
    assert result5.dtype != sample_arrays["3dgray"].dtype
    assert result6.dtype != sample_arrays["3drgb"].dtype


def test_dog(sample_arrays) -> None:

    result1: np.ndarray = pass_band.dog(sample_arrays["2dgray"])
    result2: np.ndarray = pass_band.dog(sample_arrays["3dgray"])
    result3: np.ndarray = pass_band.dog(sample_arrays["2drgb"])
    result4: np.ndarray = pass_band.dog(sample_arrays["3drgb"])
    result5: np.ndarray = pass_band.dog(sample_arrays["3dgray"], Perform_Slicewise=True)
    result6: np.ndarray = pass_band.dog(sample_arrays["3drgb"], Perform_Slicewise=True)
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape
    assert result6.shape == sample_arrays["3drgb"].shape
    assert result1.dtype != sample_arrays["2dgray"].dtype
    assert result2.dtype != sample_arrays["3dgray"].dtype
    assert result3.dtype != sample_arrays["2drgb"].dtype
    assert result4.dtype != sample_arrays["3drgb"].dtype
    assert result5.dtype != sample_arrays["3dgray"].dtype
    assert result6.dtype != sample_arrays["3drgb"].dtype


def test_gab(sample_arrays) -> None:

    result1: np.ndarray = pass_band.gab(sample_arrays["2dgray"])
    result2: np.ndarray = pass_band.gab(sample_arrays["3dgray"])
    result3: np.ndarray = pass_band.gab(sample_arrays["2drgb"])
    result4: np.ndarray = pass_band.gab(sample_arrays["3drgb"])
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
