"""
Tests for PyDoug's util module.
"""

from pathlib import Path

import numpy as np

from pydoug_core import util


def test_is3d_gray_returns(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> None:

    assert not util.is3d(sample_2dgray_array)
    assert util.is3d(sample_3dgray_array)
    assert not util.is3d(sample_2drgb_array)
    assert util.is3d(sample_3drgb_array)
    assert not util.is3d(imshape=sample_2dgray_array.shape)
    assert util.is3d(imshape=sample_3dgray_array.shape)
    assert not util.is3d(imshape=sample_2drgb_array.shape)
    assert util.is3d(imshape=sample_3drgb_array.shape)
    assert not util.is3d()


def test_isrgb_returns(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> None:

    assert not util.isrgb(sample_2dgray_array)
    assert not util.isrgb(sample_3dgray_array)
    assert util.isrgb(sample_2drgb_array)
    assert util.isrgb(sample_3drgb_array)
    assert not util.isrgb(imshape=sample_2dgray_array.shape)
    assert not util.isrgb(imshape=sample_3dgray_array.shape)
    assert util.isrgb(imshape=sample_2drgb_array.shape)
    assert util.isrgb(imshape=sample_3drgb_array.shape)
    assert not util.isrgb()


def test_is_square(sample_2dgray_array, sample_3dgray_array):

    assert util.is_square(sample_2dgray_array)
    assert not util.is_square(sample_2dgray_array[:, :-1])
    assert util.is_square(sample_3dgray_array)
    assert not util.is_square(sample_3dgray_array[:, :, :-1])
    assert util.is_square(sample_2dgray_array, False)
    assert not util.is_square(sample_2dgray_array[:, :-1], False)
    assert util.is_square(
        np.append(sample_3dgray_array, sample_3dgray_array, axis=0), False
    )
    assert not util.is_square(sample_3dgray_array, False)


def test_gray2rgb_returns(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> None:

    assert util.isrgb(util.gray2rgb(sample_2dgray_array))
    assert util.isrgb(util.gray2rgb(sample_3dgray_array))
    assert np.all(util.gray2rgb(sample_2drgb_array) == sample_2drgb_array)
    assert np.all(util.gray2rgb(sample_3drgb_array) == sample_3drgb_array)


def test_rgb2gray_returns(sample_arrays) -> None:

    assert util.rgb2gray(sample_arrays["2drgb"]).shape == sample_arrays["2dgray"].shape
    assert util.rgb2gray(sample_arrays["3drgb"]).shape == sample_arrays["3dgray"].shape
    assert util.rgb2gray(sample_arrays["2drgb"]).dtype == sample_arrays["2dgray"].dtype
    assert (
        util.rgb2gray(sample_arrays["2drgb"], False).dtype
        != sample_arrays["2dgray"].dtype
    )


def test_chan_axis_returns(sample_arrays) -> None:

    assert util.chan_axis(sample_arrays["2dgray"]) is None
    assert util.chan_axis(sample_arrays["3dgray"]) is None
    assert util.chan_axis(sample_arrays["2drgb"]) == 2
    assert util.chan_axis(sample_arrays["3drgb"]) == 3


def test_convert_type(sample_3dgray_array) -> None:

    result1: np.ndarray = util.convert_dtype(sample_3dgray_array, "Uint8")
    result2: np.ndarray = util.convert_dtype(sample_3dgray_array, "Uint16")
    result3: np.ndarray = util.convert_dtype(sample_3dgray_array, "Int16")
    result4: np.ndarray = util.convert_dtype(sample_3dgray_array, "Float32")
    result5: np.ndarray = util.convert_dtype(sample_3dgray_array, "Float64")
    result6: np.ndarray = util.convert_dtype(sample_3dgray_array, "Bool")
    uint16_array: np.ndarray = np.astype(sample_3dgray_array, np.uint16)
    uint16_array[0, 0, 0] = 65535
    result7: np.ndarray = util.convert_dtype(uint16_array, "Uint8")
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint16
    assert result3.dtype == np.int16
    assert result4.dtype == np.float32
    assert result5.dtype == np.float64
    assert result6.dtype == np.bool
    assert result7.dtype == np.uint8


def test_output_dtype_returns(sample_2dgray_array) -> None:

    assert util.output_dtype(sample_2dgray_array) == sample_2dgray_array.dtype
    assert util.output_dtype(sample_2dgray_array, preserve_dtype=False) == np.dtype(
        np.float64
    )
    assert (
        util.output_dtype(sample_2dgray_array, np.dtype(np.bool))
        == sample_2dgray_array.dtype
    )
    assert util.output_dtype(sample_2dgray_array, np.dtype(np.bool), False) == np.dtype(
        np.bool
    )


def test_axstr2int_returns(sample_arrays) -> None:

    assert util.axstr2int(sample_arrays["2dgray"], "X") == 1
    assert util.axstr2int(sample_arrays["2dgray"], "x") == 1
    assert util.axstr2int(sample_arrays["2dgray"], "y") == 0
    assert util.axstr2int(sample_arrays["2dgray"], "z") == 2
    assert util.axstr2int(sample_arrays["2dgray"], "c") == 2
    assert util.axstr2int(sample_arrays["2dgray"], "f") == 2
    assert util.axstr2int(sample_arrays["3dgray"], "X") == 2
    assert util.axstr2int(sample_arrays["3dgray"], "y") == 1
    assert util.axstr2int(sample_arrays["3dgray"], "z") == 0
    assert util.axstr2int(sample_arrays["3dgray"], "c") == 3
    assert util.axstr2int(sample_arrays["3dgray"], "f") == 3
    assert util.axstr2int(sample_arrays["2drgb"], "Z") == 3
    assert util.axstr2int(sample_arrays["3drgb"], "Z") == 0
    assert util.axstr2int(sample_arrays["2drgb"], "c") == 2
    assert util.axstr2int(sample_arrays["3drgb"], "c") == 3
    assert util.axstr2int(sample_arrays["3drgb"], "f") == 4


def test_axint2str_returns(sample_arrays) -> None:

    assert util.axint2str(sample_arrays["2dgray"], 0) == "Y"
    assert util.axint2str(sample_arrays["2dgray"], 1) == "X"
    assert util.axint2str(sample_arrays["2dgray"], 2) == "C"
    assert util.axint2str(sample_arrays["2dgray"], 3) == "C"
    assert util.axint2str(sample_arrays["2dgray"], 4) == "C"
    assert util.axint2str(sample_arrays["3dgray"], 0) == "Z"
    assert util.axint2str(sample_arrays["3dgray"], 1) == "Y"
    assert util.axint2str(sample_arrays["3dgray"], 2) == "X"
    assert util.axint2str(sample_arrays["3dgray"], 3) == "C"
    assert util.axint2str(sample_arrays["3dgray"], 4) == "C"
    assert util.axint2str(sample_arrays["2drgb"], 0) == "Y"
    assert util.axint2str(sample_arrays["3drgb"], 0) == "Z"
    assert util.axint2str(sample_arrays["2drgb"], 2) == "C"
    assert util.axint2str(sample_arrays["3drgb"], 3) == "C"


def test_axlengths_returns(sample_arrays) -> None:

    assert (
        util.axlengths(sample_arrays["2dgray"])["x"] == sample_arrays["2dgray"].shape[1]
    )
    assert (
        util.axlengths(sample_arrays["2dgray"])["y"] == sample_arrays["2dgray"].shape[0]
    )
    assert util.axlengths(sample_arrays["2dgray"])["z"] is None
    assert (
        util.axlengths(sample_arrays["2drgb"])["x"] == sample_arrays["2drgb"].shape[1]
    )
    assert (
        util.axlengths(sample_arrays["2drgb"])["y"] == sample_arrays["2drgb"].shape[0]
    )
    assert util.axlengths(sample_arrays["2drgb"])["z"] is None
    assert (
        util.axlengths(sample_arrays["3dgray"])["x"] == sample_arrays["3dgray"].shape[2]
    )
    assert (
        util.axlengths(sample_arrays["3dgray"])["y"] == sample_arrays["3dgray"].shape[1]
    )
    assert (
        util.axlengths(sample_arrays["3dgray"])["z"] == sample_arrays["3dgray"].shape[0]
    )
    assert (
        util.axlengths(sample_arrays["3drgb"])["x"] == sample_arrays["3drgb"].shape[2]
    )
    assert (
        util.axlengths(sample_arrays["3drgb"])["y"] == sample_arrays["3drgb"].shape[1]
    )
    assert (
        util.axlengths(sample_arrays["3drgb"])["z"] == sample_arrays["3drgb"].shape[0]
    )


def test_tidypath_returns(tmp_path) -> None:

    result1: Path = util.tidypath(tmp_path / "tidytest")
    result2: Path = util.tidypath(tmp_path / "tidytest.")
    result3: Path = util.tidypath(tmp_path / "tidytest..")
    result4: Path = util.tidypath(tmp_path / "tidytest...")
    result5: Path = util.tidypath(tmp_path / "tidytest", "json")
    result6: Path = util.tidypath(tmp_path / "tidytest", ".json")
    result7: Path = util.tidypath(tmp_path / "tidytest", "..json")
    result8: Path = util.tidypath(tmp_path / "tidytest", "...json")
    result9: Path = util.tidypath(tmp_path / "tidytest...", "...json")
    assert result1 == tmp_path / "tidytest.json"
    assert result2 == tmp_path / "tidytest.json"
    assert result3 == tmp_path / "tidytest.json"
    assert result4 == tmp_path / "tidytest.json"
    assert result5 == tmp_path / "tidytest.json"
    assert result6 == tmp_path / "tidytest.json"
    assert result7 == tmp_path / "tidytest.json"
    assert result8 == tmp_path / "tidytest.json"
    assert result9 == tmp_path / "tidytest.json"
    result10: Path = util.tidypath(tmp_path / "tidytest", "dir")
    result11: Path = util.tidypath(tmp_path / "tidytest.", "dir")
    result12: Path = util.tidypath(tmp_path / "tidytest..", "dir")
    result13: Path = util.tidypath(tmp_path / "tidytest...", "dir")
    result14: Path = util.tidypath(tmp_path / "tidytest", ".dir")
    result15: Path = util.tidypath(tmp_path / "tidytest", "..dir")
    result16: Path = util.tidypath(tmp_path / "tidytest", "...dir")
    result17: Path = util.tidypath(tmp_path / "tidytest...", "...dir")
    assert result10 == tmp_path / "tidytest"
    assert result11 == tmp_path / "tidytest"
    assert result12 == tmp_path / "tidytest"
    assert result13 == tmp_path / "tidytest"
    assert result14 == tmp_path / "tidytest"
    assert result15 == tmp_path / "tidytest"
    assert result16 == tmp_path / "tidytest"
    assert result17 == tmp_path / "tidytest"


def test_prep_mask(sample_arrays) -> None:

    maskarray: np.ndarray = np.copy(sample_arrays["3dgray"])
    maskarray[maskarray < 128] = 0
    maskarray[maskarray >= 128] = 255
    maskarrayrgb = np.stack([maskarray] * 3, 3)
    maskarraybool = np.astype(maskarray, np.bool)

    result0: np.ndarray = util.prep_mask(sample_arrays["2dgray"], maskarraybool[0])
    result1: np.ndarray = util.prep_mask(sample_arrays["2dgray"], maskarray[0])
    result2: np.ndarray = util.prep_mask(sample_arrays["2dgray"], maskarray)
    result3: np.ndarray = util.prep_mask(sample_arrays["2dgray"], maskarrayrgb[0])
    result4: np.ndarray = util.prep_mask(sample_arrays["2dgray"], maskarrayrgb)
    assert result0.dtype == np.dtype(np.bool)
    assert result1.dtype == np.dtype(np.bool)
    assert result2.dtype == np.dtype(np.bool)
    assert result3.dtype == np.dtype(np.bool)
    assert result4.dtype == np.dtype(np.bool)
    assert result0.shape == sample_arrays["2dgray"].shape
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2dgray"].shape
    assert result3.shape == sample_arrays["2dgray"].shape
    assert result4.shape == sample_arrays["2dgray"].shape

    result5: np.ndarray = util.prep_mask(sample_arrays["3dgray"], maskarray[0])
    result6: np.ndarray = util.prep_mask(sample_arrays["3dgray"], maskarray)
    result7: np.ndarray = util.prep_mask(sample_arrays["3dgray"], maskarrayrgb[0])
    result8: np.ndarray = util.prep_mask(sample_arrays["3dgray"], maskarrayrgb)
    assert result5.dtype == np.dtype(np.bool)
    assert result6.dtype == np.dtype(np.bool)
    assert result7.dtype == np.dtype(np.bool)
    assert result8.dtype == np.dtype(np.bool)
    assert result5.shape == sample_arrays["3dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3dgray"].shape

    result9: np.ndarray = util.prep_mask(sample_arrays["2drgb"], maskarray[0])
    result10: np.ndarray = util.prep_mask(sample_arrays["2drgb"], maskarray)
    result11: np.ndarray = util.prep_mask(sample_arrays["2drgb"], maskarrayrgb[0])
    result12: np.ndarray = util.prep_mask(sample_arrays["2drgb"], maskarrayrgb)
    assert result9.dtype == np.dtype(np.bool)
    assert result10.dtype == np.dtype(np.bool)
    assert result11.dtype == np.dtype(np.bool)
    assert result12.dtype == np.dtype(np.bool)
    assert result9.shape == sample_arrays["2drgb"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["2drgb"].shape
    assert result12.shape == sample_arrays["2drgb"].shape

    result13: np.ndarray = util.prep_mask(sample_arrays["3drgb"], maskarray[0])
    result14: np.ndarray = util.prep_mask(sample_arrays["3drgb"], maskarray)
    result15: np.ndarray = util.prep_mask(sample_arrays["3drgb"], maskarrayrgb[0])
    result16: np.ndarray = util.prep_mask(sample_arrays["3drgb"], maskarrayrgb)
    assert result13.dtype == np.dtype(np.bool)
    assert result14.dtype == np.dtype(np.bool)
    assert result15.dtype == np.dtype(np.bool)
    assert result16.dtype == np.dtype(np.bool)
    assert result13.shape == sample_arrays["3drgb"].shape
    assert result14.shape == sample_arrays["3drgb"].shape
    assert result15.shape == sample_arrays["3drgb"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_dtype_2_str() -> None:

    result1: str = util.dtype_2_str(np.dtype(np.uint8))
    result2: str = util.dtype_2_str(np.dtype(np.uint16))
    result3: str = util.dtype_2_str(np.dtype(np.int16))
    result4: str = util.dtype_2_str(np.dtype(np.float32))
    result5: str = util.dtype_2_str(np.dtype(np.float64))
    result6: str = util.dtype_2_str(np.dtype(np.bool))
    result7: str = util.dtype_2_str(np.dtype(np.int32))
    assert result1 == "Uint8"
    assert result2 == "Uint16"
    assert result3 == "Int16"
    assert result4 == "Float32"
    assert result5 == "Float64"
    assert result6 == "Bool"
    assert result7 == "Float64"
