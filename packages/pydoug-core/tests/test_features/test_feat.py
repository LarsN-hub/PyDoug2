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


def test_corners(sample_arrays) -> None:

    result0: np.ndarray = feat.corners(sample_arrays["2drgb"])
    result1: np.ndarray = feat.corners(sample_arrays["2dgray"])
    result2: np.ndarray = feat.corners(sample_arrays["3dgray"])
    result3: np.ndarray = feat.corners(sample_arrays["2dgray"], "FAST")
    result4: np.ndarray = feat.corners(sample_arrays["3dgray"], "FAST")
    result5: np.ndarray = feat.corners(sample_arrays["2dgray"], "Kitchen-Rosenfeld")
    result6: np.ndarray = feat.corners(sample_arrays["3dgray"], "Kitchen-Rosenfeld")
    result7: np.ndarray = feat.corners(sample_arrays["2dgray"], "Moravec")
    result8: np.ndarray = feat.corners(sample_arrays["3dgray"], "Moravec")
    result9: np.ndarray = feat.corners(sample_arrays["2dgray"], "Shi-Tomasi")
    result10: np.ndarray = feat.corners(sample_arrays["3dgray"], "Shi-Tomasi")
    result11: np.ndarray = feat.corners(
        sample_arrays["2dgray"], Harris_Method="Normalization"
    )
    result12: np.ndarray = feat.corners(sample_arrays["2dgray"], Return_Peaks=False)
    result13: np.ndarray = feat.corners(sample_arrays["3dgray"], Return_Peaks=False)
    result14: np.ndarray = feat.corners(sample_arrays["3dgray"], Absolute_Threshold=128)
    result15: np.ndarray = feat.corners(sample_arrays["3dgray"], Relative_Threshold=10)
    result16: np.ndarray = feat.corners(
        sample_arrays["3dgray"], Absolute_Threshold=128, Relative_Threshold=10
    )
    result17: np.ndarray = feat.corners(sample_arrays["2dgray"], Radius=1)
    result18: np.ndarray = feat.corners(sample_arrays["3dgray"], Radius=1)

    assert np.all(result0 == sample_arrays["2drgb"])
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
    assert result12.dtype == np.float64
    assert result13.dtype == np.float64
    assert result14.dtype == np.bool
    assert result15.dtype == np.bool
    assert result16.dtype == np.bool
    assert result17.dtype == np.bool
    assert result18.dtype == np.bool
