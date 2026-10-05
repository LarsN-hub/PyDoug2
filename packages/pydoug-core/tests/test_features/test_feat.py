"""
Tests for PyDoug's features.feat module.
"""

import numpy as np

from pydoug_core.features import feat


def test_peaks(sample_arrays) -> None:

    result1: np.ndarray = feat.peaks(sample_arrays["2dgray"])
    result2: np.ndarray = feat.peaks(sample_arrays["3dgray"])
    result3: np.ndarray = feat.peaks(sample_arrays["2drgb"])
    result4: np.ndarray = feat.peaks(sample_arrays["3drgb"])
    result5: np.ndarray = feat.peaks(sample_arrays["2dgray"], Perform_Slicewise=True)
    result6: np.ndarray = feat.peaks(sample_arrays["3dgray"], Perform_Slicewise=True)
    result7: np.ndarray = feat.peaks(sample_arrays["2drgb"], Perform_Slicewise=True)
    result8: np.ndarray = feat.peaks(sample_arrays["3drgb"], Perform_Slicewise=True)
    result9: np.ndarray = feat.peaks(sample_arrays["2dgray"], Absolute_Threshold=128)
    result10: np.ndarray = feat.peaks(sample_arrays["2dgray"], Relative_Threshold=10)
    result11: np.ndarray = feat.peaks(
        sample_arrays["2dgray"], Absolute_Threshold=128, Relative_Threshold=10
    )
    result12: np.ndarray = feat.peaks(sample_arrays["2dgray"], Radius=1)
    result13: np.ndarray = feat.peaks(sample_arrays["3dgray"], Radius=1)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool
    assert result5.dtype == np.bool
    assert result6.dtype == np.bool
    assert result7.dtype == np.bool
    assert result8.dtype == np.bool
    assert result9.dtype == np.bool
    assert result10.dtype == np.bool
    assert result11.dtype == np.bool
    assert result12.dtype == np.bool
    assert result13.dtype == np.bool
    assert np.all(result1 == result5)
    assert np.all(result3 == result7)
    assert np.all(result10 == result11)
