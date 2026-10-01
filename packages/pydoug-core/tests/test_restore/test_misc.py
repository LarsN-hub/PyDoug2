"""
Tests for PyDoug's restore.misc module.
"""

import numpy as np

from pydoug_core.restore import misc


def test_rolling(sample_arrays) -> None:

    result1: np.ndarray = misc.rolling(sample_arrays["2dgray"], 3)
    result2: np.ndarray = misc.rolling(sample_arrays["3dgray"], 3)
    result3: np.ndarray = misc.rolling(sample_arrays["2drgb"], 3)
    result4: np.ndarray = misc.rolling(sample_arrays["3drgb"], 3)
    result5: np.ndarray = misc.rolling(sample_arrays["2dgray"], 3, True)
    result6: np.ndarray = misc.rolling(sample_arrays["3dgray"], 3, True)
    result7: np.ndarray = misc.rolling(sample_arrays["2drgb"], 3, True)
    result8: np.ndarray = misc.rolling(sample_arrays["3drgb"], 3, True)
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["2dgray"].dtype
    assert result6.dtype == sample_arrays["3dgray"].dtype
    assert result7.dtype == sample_arrays["2drgb"].dtype
    assert result8.dtype == sample_arrays["3drgb"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["2drgb"].shape
    assert result8.shape == sample_arrays["3drgb"].shape


def test_inpaint(sample_arrays, sample_masks) -> None:

    result1: np.ndarray = misc.inpaint(sample_arrays["2dgray"], sample_masks["2dgray"])
    result2: np.ndarray = misc.inpaint(sample_arrays["3dgray"], sample_masks["3dgray"])
    result3: np.ndarray = misc.inpaint(sample_arrays["2drgb"], sample_masks["2drgb"])
    result4: np.ndarray = misc.inpaint(sample_arrays["3drgb"], sample_masks["3drgb"])
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
