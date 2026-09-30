"""
Tests for PyDoug's manipulate.crop module.
"""

import numpy as np
from skimage.morphology import disk

from pydoug_core.manipulate import crop
from pydoug_core.util import prep_mask


def test_crop(sample_arrays) -> None:

    mask_2d: np.ndarray | tuple = disk(3)
    assert isinstance(mask_2d, np.ndarray)
    mask_2d = np.pad(
        mask_2d,
        (
            (sample_arrays["2dgray"].shape[0] - mask_2d.shape[0], 0),
            (sample_arrays["2dgray"].shape[1] - mask_2d.shape[1], 0),
        ),
    )
    mask_3d: np.ndarray = prep_mask(sample_arrays["3dgray"], mask_2d)
    copy_2d1: np.ndarray = np.copy(sample_arrays["2dgray"])
    copy_2d2: np.ndarray = np.copy(sample_arrays["2drgb"])
    copy_3d1: np.ndarray = np.copy(sample_arrays["3dgray"])
    copy_3d2: np.ndarray = np.copy(sample_arrays["3dgray"])
    copy_3d3: np.ndarray = np.copy(sample_arrays["3drgb"])
    copy_3d4: np.ndarray = np.copy(sample_arrays["3drgb"])

    result1: np.ndarray = crop.crop(sample_arrays["2dgray"], mask_2d)
    result2: np.ndarray = crop.crop(sample_arrays["3dgray"], mask_2d)
    result3: np.ndarray = crop.crop(sample_arrays["2drgb"], mask_2d)
    result4: np.ndarray = crop.crop(sample_arrays["3drgb"], mask_2d)
    result5: np.ndarray = crop.crop(sample_arrays["3dgray"], mask_3d)
    result6: np.ndarray = crop.crop(sample_arrays["3drgb"], mask_3d)
    result7: np.ndarray = crop.crop(copy_2d1, mask_2d, In_Place=True)
    result8: np.ndarray = crop.crop(copy_3d1, mask_2d, In_Place=True)
    result9: np.ndarray = crop.crop(copy_2d2, mask_2d, In_Place=True)
    result10: np.ndarray = crop.crop(copy_3d3, mask_2d, In_Place=True)
    result11: np.ndarray = crop.crop(copy_3d2, mask_3d, In_Place=True)
    result12: np.ndarray = crop.crop(copy_3d4, mask_3d, In_Place=True)

    assert result1.shape != sample_arrays["2dgray"].shape
    assert result2.shape != sample_arrays["3dgray"].shape
    assert result3.shape != sample_arrays["2drgb"].shape
    assert result4.shape != sample_arrays["3drgb"].shape
    assert result5.shape != sample_arrays["3dgray"].shape
    assert result6.shape != sample_arrays["3drgb"].shape
    assert result7.shape != sample_arrays["2dgray"].shape
    assert result8.shape != sample_arrays["3dgray"].shape
    assert result9.shape != sample_arrays["2drgb"].shape
    assert result10.shape != sample_arrays["3drgb"].shape
    assert result11.shape != sample_arrays["3dgray"].shape
    assert result12.shape != sample_arrays["3drgb"].shape

    assert result1.shape == (7, 7)
    assert result2.shape == (5, 7, 7)
    assert result3.shape[:-1] == (7, 7)
    assert result4.shape[:-1] == (5, 7, 7)
    assert result5.shape == (5, 7, 7)
    assert result6.shape[:-1] == (5, 7, 7)
    assert result7.shape == result1.shape
    assert result8.shape == result2.shape
    assert result9.shape == result3.shape
    assert result10.shape == result4.shape
    assert result11.shape == result5.shape
    assert result12.shape == result6.shape

    result13: np.ndarray = crop.crop(
        sample_arrays["2dgray"], np.ones(sample_arrays["2dgray"].shape)
    )
    assert np.all(result13 == sample_arrays["2dgray"])

    single_slice_mask: np.ndarray = np.zeros(sample_arrays["3dgray"].shape, np.bool)
    single_slice_mask[3, :, :] = 1
    result14: np.ndarray = crop.crop(sample_arrays["3dgray"], single_slice_mask)
    assert result14.ndim == 2
