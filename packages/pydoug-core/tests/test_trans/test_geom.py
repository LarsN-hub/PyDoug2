"""
Tests for PyDoug's trans module.
"""

import math

import numpy as np

from pydoug_core.trans import geom


def test_mirror_returns(sample_arrays) -> None:

    result0: np.ndarray = geom.mirror(sample_arrays["2dgray"], 2)
    assert np.all(result0 == sample_arrays["2dgray"])
    result1: np.ndarray = geom.mirror(sample_arrays["2dgray"])
    result2: np.ndarray = geom.mirror(sample_arrays["3dgray"])
    result3: np.ndarray = geom.mirror(sample_arrays["2drgb"])
    result4: np.ndarray = geom.mirror(sample_arrays["3drgb"])
    assert np.all(result1[0, :] == sample_arrays["2dgray"][-1, :])
    assert np.all(result2[0] == sample_arrays["3dgray"][-1])
    assert np.all(result3[0, :] == sample_arrays["2drgb"][-1, :])
    assert np.all(result4[0] == sample_arrays["3drgb"][-1])
    result5: np.ndarray = geom.mirror(sample_arrays["2dgray"], 1)
    result6: np.ndarray = geom.mirror(sample_arrays["3dgray"], 1)
    result7: np.ndarray = geom.mirror(sample_arrays["2drgb"], 1)
    result8: np.ndarray = geom.mirror(sample_arrays["3drgb"], 1)
    assert np.all(result5[:, 0] == sample_arrays["2dgray"][:, -1])
    assert np.all(result6[:, 0, :] == sample_arrays["3dgray"][:, -1, :])
    assert np.all(result7[:, 0] == sample_arrays["2drgb"][:, -1])
    assert np.all(result8[:, 0, :] == sample_arrays["3drgb"][:, -1, :])
    result9: np.ndarray = geom.mirror(sample_arrays["3dgray"], 2)
    result10: np.ndarray = geom.mirror(sample_arrays["3drgb"], 2)
    assert np.all(result9[:, :, 0] == sample_arrays["3dgray"][:, :, -1])
    assert np.all(result10[:, :, 0] == sample_arrays["3drgb"][:, :, -1])


def test_reslice_returns(sample_arrays) -> None:

    result0: np.ndarray = geom.reslice(sample_arrays["2dgray"])
    assert np.all(result0 == sample_arrays["2dgray"])
    result01: np.ndarray = geom.reslice(sample_arrays["3dgray"], "test")
    assert np.all(result01 == sample_arrays["3dgray"])
    result1: np.ndarray = geom.reslice(sample_arrays["3dgray"])
    result2: np.ndarray = geom.reslice(sample_arrays["3drgb"])
    assert np.all(result1[0, 0, :] == sample_arrays["3dgray"][-1, 0, :])
    assert np.all(result2[0, 0, :] == sample_arrays["3drgb"][-1, 0, :])
    result3: np.ndarray = geom.reslice(sample_arrays["3dgray"], "bottom")
    result4: np.ndarray = geom.reslice(sample_arrays["3drgb"], "bottom")
    assert np.all(result3[0, 0, :] == sample_arrays["3dgray"][0, -1, :])
    assert np.all(result4[0, 0, :] == sample_arrays["3drgb"][0, -1, :])
    result5: np.ndarray = geom.reslice(sample_arrays["3dgray"], "left")
    result6: np.ndarray = geom.reslice(sample_arrays["3drgb"], "left")
    assert np.all(result5[0, 0, :] == np.flip(sample_arrays["3dgray"][:, 0, 0]))
    assert np.all(result6[0, 0, :] == np.flip(sample_arrays["3drgb"][:, 0, 0], 0))
    result7: np.ndarray = geom.reslice(sample_arrays["3dgray"], "right")
    result8: np.ndarray = geom.reslice(sample_arrays["3drgb"], "right")
    assert np.all(result7[0, 0, :] == sample_arrays["3dgray"][:, 0, -1])
    assert np.all(result8[0, 0, :] == sample_arrays["3drgb"][:, 0, -1])
    result9: np.ndarray = geom.reslice(sample_arrays["3dgray"], "back")
    result10: np.ndarray = geom.reslice(sample_arrays["3drgb"], "back")
    assert np.all(result9[0, 0, :] == np.flip(sample_arrays["3dgray"][-1, 0, :]))
    assert np.all(result10[0, 0, :] == np.flip(sample_arrays["3drgb"][-1, 0, :], 0))


def test_rotate_returns(sample_arrays) -> None:

    result0: np.ndarray = geom.rotate(sample_arrays["2dgray"], 0)
    assert np.all(result0 == sample_arrays["2dgray"])
    mirarray: np.ndarray = np.vstack(
        (sample_arrays["2dgray"], np.flipud(sample_arrays["2dgray"]))
    )
    result1: np.ndarray = geom.rotate(mirarray, 10)
    result2: np.ndarray = geom.rotate(mirarray, 10, True)
    assert np.all(result1 == np.flipud(result2))
    assert result1.dtype == sample_arrays["2dgray"].dtype
    result3: np.ndarray = geom.rotate(sample_arrays["2dgray"], 20, resize=True)
    assert result3.shape[0] > sample_arrays["2dgray"].shape[0]
    assert result3.shape[1] > sample_arrays["2dgray"].shape[1]
    result4: np.ndarray = geom.rotate(sample_arrays["2dgray"], 30, preserve_dtype=False)
    assert result4.dtype == np.dtype(np.float64)
    result5: np.ndarray = geom.rotate(sample_arrays["3dgray"], 50, resize=True)
    assert result5.shape[0] == sample_arrays["3dgray"].shape[0]
    result6: np.ndarray = geom.rotate(sample_arrays["3drgb"], 50, resize=True)
    assert result6.shape[0] == sample_arrays["3drgb"].shape[0]


def test_rescale_returns(sample_arrays) -> None:

    result1: np.ndarray = geom.rescale(sample_arrays["2dgray"])
    assert np.all(result1 == sample_arrays["2dgray"])
    result2: np.ndarray = geom.rescale(sample_arrays["2dgray"], 0.5)
    result3: np.ndarray = geom.rescale(
        sample_arrays["3dgray"], 1.5, preserve_dtype=False
    )
    assert result2.dtype == sample_arrays["2dgray"].dtype
    assert result2.shape[0] <= math.ceil(sample_arrays["2dgray"].shape[0] * 0.5)
    assert result2.shape[1] <= math.ceil(sample_arrays["2dgray"].shape[1] * 0.5)
    assert result3.dtype == np.dtype(np.float64)
    assert result3.shape[0] >= math.floor(sample_arrays["3dgray"].shape[0] * 1.5)
    assert result3.shape[1] >= math.floor(sample_arrays["3dgray"].shape[1] * 1.5)
    assert result3.shape[2] >= math.floor(sample_arrays["3dgray"].shape[2] * 1.5)
    result4: np.ndarray = geom.rescale(sample_arrays["3drgb"], 0.5)
    assert result4.shape[0] <= math.ceil(sample_arrays["3drgb"].shape[0] * 0.5)
    assert result4.shape[1] <= math.ceil(sample_arrays["3drgb"].shape[1] * 0.5)
    assert result4.shape[2] <= math.ceil(sample_arrays["3drgb"].shape[2] * 0.5)
    assert result4.shape[3] == sample_arrays["3drgb"].shape[3]
    result5: np.ndarray = geom.rescale(
        sample_arrays["2dgray"] >= 128, 0.5, segmented_data=True
    )
    assert result5.dtype == np.dtype(np.bool)
    result6: np.ndarray = geom.rescale(sample_arrays["2dgray"] >= 128, 0.5)
    assert result6.dtype == np.dtype(np.bool)


def test_resize_returns(sample_arrays) -> None:

    result0: np.ndarray = geom.resize(sample_arrays["2dgray"], 20, 50)
    result1: np.ndarray = geom.resize(sample_arrays["3dgray"])
    result2: np.ndarray = geom.resize(sample_arrays["3dgray"], 10, 10)
    result3: np.ndarray = geom.resize(sample_arrays["3dgray"], 10, 10, 5)
    result4: np.ndarray = geom.resize(sample_arrays["3dgray"], 12, 12, 5)
    result5: np.ndarray = geom.resize(sample_arrays["3dgray"], 12, 12, 6)
    result6: np.ndarray = geom.resize(sample_arrays["3drgb"], 12, 12, 6)
    result7: np.ndarray = geom.resize(sample_arrays["3dgray"], 8, 8, 4)
    result8: np.ndarray = geom.resize(
        sample_arrays["3dgray"], 8, 8, 4, preserve_dtype=False
    )
    result9: np.ndarray = geom.resize(sample_arrays["3dgray"] >= 128, 8, 8, 4)
    result10: np.ndarray = geom.resize(
        sample_arrays["3dgray"] >= 128, 8, 8, 4, segmented_data=True
    )
    assert result0.shape == (50, 20)
    assert np.all(result1 == sample_arrays["3dgray"])
    assert np.all(result2 == sample_arrays["3dgray"])
    assert np.all(result3 == sample_arrays["3dgray"])
    assert result4.shape == (5, 12, 12)
    assert result5.shape == (6, 12, 12)
    assert result6.shape == (6, 12, 12, 3)
    assert result7.shape == (4, 8, 8)
    assert result8.shape == (4, 8, 8)
    assert result8.dtype == np.dtype(np.float64)
    assert result9.dtype == np.dtype(np.bool)
    assert result10.dtype == np.dtype(np.bool)
