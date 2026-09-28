"""
Tests for PyDoug's filter.blur module.
"""

import numpy as np

from pydoug_core.filter import blur


def test_smooth(sample_arrays) -> None:

    result1: np.ndarray = blur.smooth(sample_arrays["2dgray"])
    result2: np.ndarray = blur.smooth(sample_arrays["3dgray"])
    result3: np.ndarray = blur.smooth(sample_arrays["2drgb"])
    result4: np.ndarray = blur.smooth(sample_arrays["3drgb"])
    result5: np.ndarray = blur.smooth(sample_arrays["2dgray"], Method="Median")
    result6: np.ndarray = blur.smooth(sample_arrays["3dgray"], Method="Median")
    result7: np.ndarray = blur.smooth(sample_arrays["2drgb"], Method="Median")
    result8: np.ndarray = blur.smooth(sample_arrays["3drgb"], Method="Median")
    result9: np.ndarray = blur.smooth(sample_arrays["2dgray"], Method="Mean")
    result10: np.ndarray = blur.smooth(sample_arrays["3dgray"], Method="Mean")
    result11: np.ndarray = blur.smooth(sample_arrays["2drgb"], Method="Mean")
    result12: np.ndarray = blur.smooth(sample_arrays["3drgb"], Method="Mean")
    result13: np.ndarray = blur.smooth(sample_arrays["2dgray"], Method="Mode")
    result14: np.ndarray = blur.smooth(sample_arrays["3dgray"], Method="Mode")
    result15: np.ndarray = blur.smooth(sample_arrays["2drgb"], Method="Mode")
    result16: np.ndarray = blur.smooth(sample_arrays["3drgb"], Method="Mode")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["2drgb"].shape
    assert result8.shape == sample_arrays["3drgb"].shape
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["3dgray"].shape
    assert result11.shape == sample_arrays["2drgb"].shape
    assert result12.shape == sample_arrays["3drgb"].shape
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["3dgray"].shape
    assert result15.shape == sample_arrays["2drgb"].shape
    assert result16.shape == sample_arrays["3drgb"].shape
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["2dgray"].dtype
    assert result6.dtype == sample_arrays["3dgray"].dtype
    assert result7.dtype == sample_arrays["2drgb"].dtype
    assert result8.dtype == sample_arrays["3drgb"].dtype
    assert result9.dtype == sample_arrays["2dgray"].dtype
    assert result10.dtype == sample_arrays["3dgray"].dtype
    assert result11.dtype == sample_arrays["2drgb"].dtype
    assert result12.dtype == sample_arrays["3drgb"].dtype
    assert result13.dtype == sample_arrays["2dgray"].dtype
    assert result14.dtype == sample_arrays["3dgray"].dtype
    assert result15.dtype == sample_arrays["2drgb"].dtype
    assert result16.dtype == sample_arrays["3drgb"].dtype

    result17: np.ndarray = blur.smooth(sample_arrays["2dgray"], Preserve_Dtype=False)
    result18: np.ndarray = blur.smooth(sample_arrays["3dgray"], Preserve_Dtype=False)
    result19: np.ndarray = blur.smooth(sample_arrays["2drgb"], Preserve_Dtype=False)
    result20: np.ndarray = blur.smooth(sample_arrays["3drgb"], Preserve_Dtype=False)
    result21: np.ndarray = blur.smooth(
        sample_arrays["2dgray"], Method="Median", Preserve_Dtype=False
    )
    result22: np.ndarray = blur.smooth(
        sample_arrays["3dgray"], Method="Median", Preserve_Dtype=False
    )
    result23: np.ndarray = blur.smooth(
        sample_arrays["2drgb"], Method="Median", Preserve_Dtype=False
    )
    result24: np.ndarray = blur.smooth(
        sample_arrays["3drgb"], Method="Median", Preserve_Dtype=False
    )
    result25: np.ndarray = blur.smooth(
        sample_arrays["2dgray"], Method="Mean", Preserve_Dtype=False
    )
    result26: np.ndarray = blur.smooth(
        sample_arrays["3dgray"], Method="Mean", Preserve_Dtype=False
    )
    result27: np.ndarray = blur.smooth(
        sample_arrays["2drgb"], Method="Mean", Preserve_Dtype=False
    )
    result28: np.ndarray = blur.smooth(
        sample_arrays["3drgb"], Method="Mean", Preserve_Dtype=False
    )
    result29: np.ndarray = blur.smooth(
        sample_arrays["2dgray"], Method="Mode", Preserve_Dtype=False
    )
    result30: np.ndarray = blur.smooth(
        sample_arrays["3dgray"], Method="Mode", Preserve_Dtype=False
    )
    result31: np.ndarray = blur.smooth(
        sample_arrays["2drgb"], Method="Mode", Preserve_Dtype=False
    )
    result32: np.ndarray = blur.smooth(
        sample_arrays["3drgb"], Method="Mode", Preserve_Dtype=False
    )
    assert result17.shape == sample_arrays["2dgray"].shape
    assert result18.shape == sample_arrays["3dgray"].shape
    assert result19.shape == sample_arrays["2drgb"].shape
    assert result20.shape == sample_arrays["3drgb"].shape
    assert result21.shape == sample_arrays["2dgray"].shape
    assert result22.shape == sample_arrays["3dgray"].shape
    assert result23.shape == sample_arrays["2drgb"].shape
    assert result24.shape == sample_arrays["3drgb"].shape
    assert result25.shape == sample_arrays["2dgray"].shape
    assert result26.shape == sample_arrays["3dgray"].shape
    assert result27.shape == sample_arrays["2drgb"].shape
    assert result28.shape == sample_arrays["3drgb"].shape
    assert result29.shape == sample_arrays["2dgray"].shape
    assert result30.shape == sample_arrays["3dgray"].shape
    assert result31.shape == sample_arrays["2drgb"].shape
    assert result32.shape == sample_arrays["3drgb"].shape
    assert result17.dtype != sample_arrays["2dgray"].dtype
    assert result18.dtype != sample_arrays["3dgray"].dtype
    assert result19.dtype != sample_arrays["2drgb"].dtype
    assert result20.dtype != sample_arrays["3drgb"].dtype
    assert result21.dtype == sample_arrays["2dgray"].dtype
    assert result22.dtype == sample_arrays["3dgray"].dtype
    assert result23.dtype == sample_arrays["2drgb"].dtype
    assert result24.dtype == sample_arrays["3drgb"].dtype
    assert result25.dtype == sample_arrays["2dgray"].dtype
    assert result26.dtype == sample_arrays["3dgray"].dtype
    assert result27.dtype == sample_arrays["2drgb"].dtype
    assert result28.dtype == sample_arrays["3drgb"].dtype
    assert result29.dtype == sample_arrays["2dgray"].dtype
    assert result30.dtype == sample_arrays["3dgray"].dtype
    assert result31.dtype == sample_arrays["2drgb"].dtype
    assert result32.dtype == sample_arrays["3drgb"].dtype

    result33: np.ndarray = blur.smooth(sample_arrays["3dgray"], Perform_Slicewise=True)
    result34: np.ndarray = blur.smooth(
        sample_arrays["3dgray"], Perform_Slicewise=True, Preserve_Dtype=False
    )
    assert result33.shape == sample_arrays["3dgray"].shape
    assert result34.shape == sample_arrays["3dgray"].shape
    assert result33.dtype == sample_arrays["3dgray"].dtype
    assert result34.dtype != sample_arrays["3dgray"].dtype
