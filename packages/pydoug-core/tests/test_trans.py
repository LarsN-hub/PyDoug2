"""
Tests for PyDoug's trans module.
"""

import numpy as np

from pydoug_core import trans


def test_mirror_returns(sample_arrays) -> None:

    result0: np.ndarray = trans.mirror(sample_arrays["2dgray"], 2)
    assert np.all(result0 == sample_arrays["2dgray"])
    result1: np.ndarray = trans.mirror(sample_arrays["2dgray"])
    result2: np.ndarray = trans.mirror(sample_arrays["3dgray"])
    result3: np.ndarray = trans.mirror(sample_arrays["2drgb"])
    result4: np.ndarray = trans.mirror(sample_arrays["3drgb"])
    assert np.all(result1[0, :] == sample_arrays["2dgray"][-1, :])
    assert np.all(result2[0] == sample_arrays["3dgray"][-1])
    assert np.all(result3[0, :] == sample_arrays["2drgb"][-1, :])
    assert np.all(result4[0] == sample_arrays["3drgb"][-1])
    result5: np.ndarray = trans.mirror(sample_arrays["2dgray"], 1)
    result6: np.ndarray = trans.mirror(sample_arrays["3dgray"], 1)
    result7: np.ndarray = trans.mirror(sample_arrays["2drgb"], 1)
    result8: np.ndarray = trans.mirror(sample_arrays["3drgb"], 1)
    assert np.all(result5[:, 0] == sample_arrays["2dgray"][:, -1])
    assert np.all(result6[:, 0, :] == sample_arrays["3dgray"][:, -1, :])
    assert np.all(result7[:, 0] == sample_arrays["2drgb"][:, -1])
    assert np.all(result8[:, 0, :] == sample_arrays["3drgb"][:, -1, :])
    result9: np.ndarray = trans.mirror(sample_arrays["3dgray"], 2)
    result10: np.ndarray = trans.mirror(sample_arrays["3drgb"], 2)
    assert np.all(result9[:, :, 0] == sample_arrays["3dgray"][:, :, -1])
    assert np.all(result10[:, :, 0] == sample_arrays["3drgb"][:, :, -1])


def test_reslice_returns(sample_arrays) -> None:

    result0: np.ndarray = trans.reslice(sample_arrays["2dgray"])
    assert np.all(result0 == sample_arrays["2dgray"])
    result01: np.ndarray = trans.reslice(sample_arrays["3dgray"], "test")
    assert np.all(result01 == sample_arrays["3dgray"])
    result1: np.ndarray = trans.reslice(sample_arrays["3dgray"])
    result2: np.ndarray = trans.reslice(sample_arrays["3drgb"])
    assert np.all(result1[0, 0, :] == sample_arrays["3dgray"][-1, 0, :])
    assert np.all(result2[0, 0, :] == sample_arrays["3drgb"][-1, 0, :])
    result3: np.ndarray = trans.reslice(sample_arrays["3dgray"], "bottom")
    result4: np.ndarray = trans.reslice(sample_arrays["3drgb"], "bottom")
    assert np.all(result3[0, 0, :] == sample_arrays["3dgray"][0, -1, :])
    assert np.all(result4[0, 0, :] == sample_arrays["3drgb"][0, -1, :])
    result5: np.ndarray = trans.reslice(sample_arrays["3dgray"], "left")
    result6: np.ndarray = trans.reslice(sample_arrays["3drgb"], "left")
    assert np.all(result5[0, 0, :] == np.flip(sample_arrays["3dgray"][:, 0, 0]))
    assert np.all(result6[0, 0, :] == np.flip(sample_arrays["3drgb"][:, 0, 0], 0))
    result7: np.ndarray = trans.reslice(sample_arrays["3dgray"], "right")
    result8: np.ndarray = trans.reslice(sample_arrays["3drgb"], "right")
    assert np.all(result7[0, 0, :] == sample_arrays["3dgray"][:, 0, -1])
    assert np.all(result8[0, 0, :] == sample_arrays["3drgb"][:, 0, -1])
    result9: np.ndarray = trans.reslice(sample_arrays["3dgray"], "back")
    result10: np.ndarray = trans.reslice(sample_arrays["3drgb"], "back")
    assert np.all(result9[0, 0, :] == np.flip(sample_arrays["3dgray"][-1, 0, :]))
    assert np.all(result10[0, 0, :] == np.flip(sample_arrays["3drgb"][-1, 0, :], 0))


def test_rotate_returns(sample_arrays) -> None:

    result0: np.ndarray = trans.rotate(sample_arrays["2dgray"], 0)
    assert np.all(result0 == sample_arrays["2dgray"])
    mirarray: np.ndarray = np.vstack(
        (sample_arrays["2dgray"], np.flipud(sample_arrays["2dgray"]))
    )
    result1: np.ndarray = trans.rotate(mirarray, 10)
    result2: np.ndarray = trans.rotate(mirarray, 10, True)
    assert np.all(result1 == np.flipud(result2))
    assert result1.dtype == sample_arrays["2dgray"].dtype
    result3: np.ndarray = trans.rotate(sample_arrays["2dgray"], 20, resize=True)
    assert result3.shape[0] > sample_arrays["2dgray"].shape[0]
    assert result3.shape[1] > sample_arrays["2dgray"].shape[1]
    result4: np.ndarray = trans.rotate(
        sample_arrays["2dgray"], 30, preserve_dtype=False
    )
    assert result4.dtype == np.dtype(np.float64)
    result5: np.ndarray = trans.rotate(sample_arrays["3dgray"], 50, resize=True)
    assert result5.shape[0] == sample_arrays["3dgray"].shape[0]
    result6: np.ndarray = trans.rotate(sample_arrays["3drgb"], 50, resize=True)
    assert result6.shape[0] == sample_arrays["3drgb"].shape[0]
