"""
Tests for PyDoug's restore.denoise module.
"""

import numpy as np

from pydoug_core.restore import denoise


def test_bilateral(sample_arrays) -> None:

    result1: np.ndarray = denoise.bilateral(sample_arrays["2dgray"])
    result2: np.ndarray = denoise.bilateral(sample_arrays["3dgray"])
    result3: np.ndarray = denoise.bilateral(sample_arrays["2drgb"])
    result4: np.ndarray = denoise.bilateral(sample_arrays["3drgb"])
    result5: np.ndarray = denoise.bilateral(sample_arrays["2dgray"], Rank=True)
    result6: np.ndarray = denoise.bilateral(sample_arrays["3dgray"], Rank=True)
    result7: np.ndarray = denoise.bilateral(sample_arrays["2drgb"], Rank=True)
    result8: np.ndarray = denoise.bilateral(sample_arrays["3drgb"], Rank=True)
    result9: np.ndarray = denoise.bilateral(
        sample_arrays["2dgray"], Preserve_Dtype=False
    )
    result10: np.ndarray = denoise.bilateral(
        sample_arrays["3dgray"], Preserve_Dtype=False
    )
    result11: np.ndarray = denoise.bilateral(
        sample_arrays["2drgb"], Preserve_Dtype=False
    )
    result12: np.ndarray = denoise.bilateral(
        sample_arrays["3drgb"], Preserve_Dtype=False
    )
    result13: np.ndarray = denoise.bilateral(
        sample_arrays["2dgray"], Rank=True, Preserve_Dtype=False
    )
    result14: np.ndarray = denoise.bilateral(
        sample_arrays["3dgray"], Rank=True, Preserve_Dtype=False
    )
    result15: np.ndarray = denoise.bilateral(
        sample_arrays["2drgb"], Rank=True, Preserve_Dtype=False
    )
    result16: np.ndarray = denoise.bilateral(
        sample_arrays["3drgb"], Rank=True, Preserve_Dtype=False
    )
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.uint8
    assert result6.dtype == np.uint8
    assert result7.dtype == np.uint8
    assert result8.dtype == np.uint8
    assert result9.dtype == np.dtype(np.float64)
    assert result10.dtype == np.dtype(np.float64)
    assert result11.dtype == np.dtype(np.float64)
    assert result12.dtype == np.dtype(np.float64)
    assert result13.dtype == np.uint8
    assert result14.dtype == np.uint8
    assert result15.dtype == np.uint8
    assert result16.dtype == np.uint8
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["2drgb"].shape
    assert result8.shape == sample_arrays["3drgb"].shape


def test_nl_means(sample_arrays) -> None:

    result1: np.ndarray = denoise.nl_means(sample_arrays["2dgray"], Fast_Mode=True)
    result2: np.ndarray = denoise.nl_means(sample_arrays["3dgray"], Fast_Mode=True)
    result3: np.ndarray = denoise.nl_means(sample_arrays["2drgb"], Fast_Mode=True)
    result4: np.ndarray = denoise.nl_means(sample_arrays["3drgb"], Fast_Mode=True)
    result5: np.ndarray = denoise.nl_means(
        sample_arrays["2dgray"], Fast_Mode=True, Preserve_Dtype=False
    )
    result6: np.ndarray = denoise.nl_means(
        sample_arrays["3dgray"], Fast_Mode=True, Preserve_Dtype=False
    )
    result7: np.ndarray = denoise.nl_means(
        sample_arrays["2drgb"], Fast_Mode=True, Preserve_Dtype=False
    )
    result8: np.ndarray = denoise.nl_means(
        sample_arrays["3drgb"], Fast_Mode=True, Preserve_Dtype=False
    )
    result9: np.ndarray = denoise.nl_means(
        sample_arrays["2dgray"], Fast_Mode=True, Slicewise=True
    )
    result10: np.ndarray = denoise.nl_means(
        sample_arrays["3dgray"], Fast_Mode=True, Slicewise=True
    )
    result11: np.ndarray = denoise.nl_means(
        sample_arrays["2drgb"], Fast_Mode=True, Slicewise=True
    )
    result12: np.ndarray = denoise.nl_means(
        sample_arrays["3drgb"], Fast_Mode=True, Slicewise=True
    )
    result13: np.ndarray = denoise.nl_means(
        sample_arrays["2dgray"], Fast_Mode=True, Slicewise=True, Preserve_Dtype=False
    )
    result14: np.ndarray = denoise.nl_means(
        sample_arrays["3dgray"], Fast_Mode=True, Slicewise=True, Preserve_Dtype=False
    )
    result15: np.ndarray = denoise.nl_means(
        sample_arrays["2drgb"], Fast_Mode=True, Slicewise=True, Preserve_Dtype=False
    )
    result16: np.ndarray = denoise.nl_means(
        sample_arrays["3drgb"], Fast_Mode=True, Slicewise=True, Preserve_Dtype=False
    )
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.dtype(np.float64)
    assert result6.dtype == np.dtype(np.float64)
    assert result7.dtype == np.dtype(np.float64)
    assert result8.dtype == np.dtype(np.float64)
    assert result9.dtype == np.uint8
    assert result10.dtype == np.uint8
    assert result11.dtype == np.uint8
    assert result12.dtype == np.uint8
    assert result13.dtype == np.dtype(np.float64)
    assert result14.dtype == np.dtype(np.float64)
    assert result15.dtype == np.dtype(np.float64)
    assert result16.dtype == np.dtype(np.float64)
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


def test_tv_bregman(sample_arrays) -> None:

    result1: np.ndarray = denoise.tv_bregman(sample_arrays["2dgray"])
    result2: np.ndarray = denoise.tv_bregman(sample_arrays["3dgray"])
    result3: np.ndarray = denoise.tv_bregman(sample_arrays["2drgb"])
    result4: np.ndarray = denoise.tv_bregman(sample_arrays["3drgb"])
    result5: np.ndarray = denoise.tv_bregman(
        sample_arrays["2dgray"], Preserve_Dtype=False
    )
    result6: np.ndarray = denoise.tv_bregman(
        sample_arrays["3dgray"], Preserve_Dtype=False
    )
    result7: np.ndarray = denoise.tv_bregman(
        sample_arrays["2drgb"], Preserve_Dtype=False
    )
    result8: np.ndarray = denoise.tv_bregman(
        sample_arrays["3drgb"], Preserve_Dtype=False
    )
    result9: np.ndarray = denoise.tv_bregman(sample_arrays["2dgray"], Slicewise=True)
    result10: np.ndarray = denoise.tv_bregman(sample_arrays["3dgray"], Slicewise=True)
    result11: np.ndarray = denoise.tv_bregman(sample_arrays["2drgb"], Slicewise=True)
    result12: np.ndarray = denoise.tv_bregman(sample_arrays["3drgb"], Slicewise=True)
    result13: np.ndarray = denoise.tv_bregman(
        sample_arrays["2dgray"], Slicewise=True, Preserve_Dtype=False
    )
    result14: np.ndarray = denoise.tv_bregman(
        sample_arrays["3dgray"], Slicewise=True, Preserve_Dtype=False
    )
    result15: np.ndarray = denoise.tv_bregman(
        sample_arrays["2drgb"], Slicewise=True, Preserve_Dtype=False
    )
    result16: np.ndarray = denoise.tv_bregman(
        sample_arrays["3drgb"], Slicewise=True, Preserve_Dtype=False
    )
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.dtype(np.float64)
    assert result6.dtype == np.dtype(np.float64)
    assert result7.dtype == np.dtype(np.float64)
    assert result8.dtype == np.dtype(np.float64)
    assert result9.dtype == np.uint8
    assert result10.dtype == np.uint8
    assert result11.dtype == np.uint8
    assert result12.dtype == np.uint8
    assert result13.dtype == np.dtype(np.float64)
    assert result14.dtype == np.dtype(np.float64)
    assert result15.dtype == np.dtype(np.float64)
    assert result16.dtype == np.dtype(np.float64)
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


def test_tv_chambolle(sample_arrays) -> None:

    result1: np.ndarray = denoise.tv_chambolle(sample_arrays["2dgray"])
    result2: np.ndarray = denoise.tv_chambolle(sample_arrays["3dgray"])
    result3: np.ndarray = denoise.tv_chambolle(sample_arrays["2drgb"])
    result4: np.ndarray = denoise.tv_chambolle(sample_arrays["3drgb"])
    result5: np.ndarray = denoise.tv_chambolle(
        sample_arrays["2dgray"], Preserve_Dtype=False
    )
    result6: np.ndarray = denoise.tv_chambolle(
        sample_arrays["3dgray"], Preserve_Dtype=False
    )
    result7: np.ndarray = denoise.tv_chambolle(
        sample_arrays["2drgb"], Preserve_Dtype=False
    )
    result8: np.ndarray = denoise.tv_chambolle(
        sample_arrays["3drgb"], Preserve_Dtype=False
    )
    result9: np.ndarray = denoise.tv_chambolle(sample_arrays["2dgray"], Slicewise=True)
    result10: np.ndarray = denoise.tv_chambolle(sample_arrays["3dgray"], Slicewise=True)
    result11: np.ndarray = denoise.tv_chambolle(sample_arrays["2drgb"], Slicewise=True)
    result12: np.ndarray = denoise.tv_chambolle(sample_arrays["3drgb"], Slicewise=True)
    result13: np.ndarray = denoise.tv_chambolle(
        sample_arrays["2dgray"], Slicewise=True, Preserve_Dtype=False
    )
    result14: np.ndarray = denoise.tv_chambolle(
        sample_arrays["3dgray"], Slicewise=True, Preserve_Dtype=False
    )
    result15: np.ndarray = denoise.tv_chambolle(
        sample_arrays["2drgb"], Slicewise=True, Preserve_Dtype=False
    )
    result16: np.ndarray = denoise.tv_chambolle(
        sample_arrays["3drgb"], Slicewise=True, Preserve_Dtype=False
    )
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.dtype(np.float64)
    assert result6.dtype == np.dtype(np.float64)
    assert result7.dtype == np.dtype(np.float64)
    assert result8.dtype == np.dtype(np.float64)
    assert result9.dtype == np.uint8
    assert result10.dtype == np.uint8
    assert result11.dtype == np.uint8
    assert result12.dtype == np.uint8
    assert result13.dtype == np.dtype(np.float64)
    assert result14.dtype == np.dtype(np.float64)
    assert result15.dtype == np.dtype(np.float64)
    assert result16.dtype == np.dtype(np.float64)
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


def test_wavelet(sample_arrays) -> None:

    result1: np.ndarray = denoise.wavelet(sample_arrays["2dgray"])
    result2: np.ndarray = denoise.wavelet(sample_arrays["3dgray"])
    result3: np.ndarray = denoise.wavelet(sample_arrays["2drgb"])
    result4: np.ndarray = denoise.wavelet(sample_arrays["3drgb"])
    result5: np.ndarray = denoise.wavelet(sample_arrays["2dgray"], Preserve_Dtype=False)
    result6: np.ndarray = denoise.wavelet(sample_arrays["3dgray"], Preserve_Dtype=False)
    result7: np.ndarray = denoise.wavelet(sample_arrays["2drgb"], Preserve_Dtype=False)
    result8: np.ndarray = denoise.wavelet(sample_arrays["3drgb"], Preserve_Dtype=False)
    result9: np.ndarray = denoise.wavelet(sample_arrays["2dgray"], Slicewise=True)
    result10: np.ndarray = denoise.wavelet(sample_arrays["3dgray"], Slicewise=True)
    result11: np.ndarray = denoise.wavelet(sample_arrays["2drgb"], Slicewise=True)
    result12: np.ndarray = denoise.wavelet(sample_arrays["3drgb"], Slicewise=True)
    result13: np.ndarray = denoise.wavelet(
        sample_arrays["2dgray"], Slicewise=True, Preserve_Dtype=False
    )
    result14: np.ndarray = denoise.wavelet(
        sample_arrays["3dgray"], Slicewise=True, Preserve_Dtype=False
    )
    result15: np.ndarray = denoise.wavelet(
        sample_arrays["2drgb"], Slicewise=True, Preserve_Dtype=False
    )
    result16: np.ndarray = denoise.wavelet(
        sample_arrays["3drgb"], Slicewise=True, Preserve_Dtype=False
    )
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.dtype(np.float64)
    assert result6.dtype == np.dtype(np.float64)
    assert result7.dtype == np.dtype(np.float64)
    assert result8.dtype == np.dtype(np.float64)
    assert result9.dtype == np.uint8
    assert result10.dtype == np.uint8
    assert result11.dtype == np.uint8
    assert result12.dtype == np.uint8
    assert result13.dtype == np.dtype(np.float64)
    assert result14.dtype == np.dtype(np.float64)
    assert result15.dtype == np.dtype(np.float64)
    assert result16.dtype == np.dtype(np.float64)
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
