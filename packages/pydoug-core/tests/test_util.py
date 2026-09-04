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


def test_chan_axis_returns(sample_arrays) -> None:

    assert util.chan_axis(sample_arrays["2dgray"]) is None
    assert util.chan_axis(sample_arrays["3dgray"]) is None
    assert util.chan_axis(sample_arrays["2drgb"]) == 2
    assert util.chan_axis(sample_arrays["3drgb"]) == 3


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
