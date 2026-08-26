"""
Tests for PyDoug io.reader module.
"""

import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pydoug_core import util
from pydoug_core.io import imreader as imr


def test_hdf_expand_tree_returns(sample_single_hdf, sample_empty_hdf) -> None:

    result1 = imr._hdf_expand_tree(sample_single_hdf)
    result2 = imr._hdf_expand_tree(sample_empty_hdf)
    assert isinstance(result1, tuple)
    assert isinstance(result1[0], list)
    assert isinstance(result1[1], bool)
    assert isinstance(result2, tuple)
    assert not result2[1]
    assert isinstance(result2[1], bool)


def test_hdf_expand_tree_list(
    sample_single_hdf, sample_double_hdf, sample_group_hdf
) -> None:

    result1 = imr._hdf_expand_tree(sample_single_hdf)
    result2 = imr._hdf_expand_tree(sample_double_hdf)
    result3 = imr._hdf_expand_tree(sample_double_hdf, break_on_dset=True)
    result4 = imr._hdf_expand_tree(sample_group_hdf, printtree=True)
    assert isinstance(result1[0], list)
    assert isinstance(result2[0], list)
    assert isinstance(result3[0], list)
    assert isinstance(result4[0], list)
    assert len(result1[0]) == 1
    assert len(result2[0]) == 2
    assert len(result3[0]) == 1
    assert len(result4[0]) == 1


def test_hdf_largest_dataset_returns(sample_single_hdf) -> None:

    result = imr._hdf_largest_dataset(sample_single_hdf)
    assert isinstance(result, str)


def test_imread_tiff(sample_tiff, sample_arrays) -> None:

    path, kind = sample_tiff
    result = imr.imread(path)
    np.testing.assert_array_equal(result, sample_arrays[kind])


def test_imread_hdf(sample_hdf, sample_arrays) -> None:

    path, kind = sample_hdf
    result = imr.imread(path)
    np.testing.assert_array_equal(result, sample_arrays[kind])


def test_imread_nonexistant_file(tmp_path) -> None:

    path1: Path = Path(tmp_path / "nonexistant.tiff")
    path2: Path = Path(tmp_path / "nonexistant.hdf")
    assert not imr.imread(path1)
    assert not imr.imread(path2)


def test_imread_empty_files(sample_empty_tiff, sample_empty_hdf) -> None:

    result1 = imr.imread(sample_empty_tiff)
    result2 = imr.imread(sample_empty_hdf)
    assert not result1
    assert not result2


def test_imread_unsupported_format(sample_unsupported_format) -> None:

    assert not imr.imread(sample_unsupported_format)


def test_imread_unexpected_error(tmp_path, monkeypatch) -> None:
    def _raise_unexpected(*args, **kwargs):
        raise ValueError("Simulated value error.")

    monkeypatch.setattr(iio, "imread", _raise_unexpected)
    assert imr.imread(tmp_path) is None


def test_imread_no_input() -> None:

    assert not imr.imread(None)


def test_seqread_no_input() -> None:

    assert not imr.seqread(None)


def test_seqread_empty_input(sample_empty_dir) -> None:

    assert not imr.seqread(sample_empty_dir)


def test_seqread_nonexistant_input(tmp_path) -> None:

    assert not imr.seqread(tmp_path / "nonexistant")


def test_seqread_returns(
    sample_2dgray_seq,
    sample_3dgray_array,
    sample_2drgb_seq,
    sample_3drgb_array,
    sample_2d3dgrayrgb_seq,
    sample_misshaped_seq,
    sample_zero_size_seq,
) -> None:

    assert np.all(imr.seqread(sample_2dgray_seq) == sample_3dgray_array)
    assert np.all(imr.seqread(sample_2drgb_seq) == sample_3drgb_array)
    assert np.all(
        imr.seqread(sample_2d3dgrayrgb_seq) == util.gray2rgb(sample_3dgray_array)
    )
    assert np.all(imr.seqread(sample_2dgray_seq, 2) == sample_3dgray_array)
    assert np.all(imr.seqread(sample_2drgb_seq, None) == sample_3drgb_array)
    assert not imr.seqread(sample_misshaped_seq)
    assert not imr.seqread(sample_zero_size_seq)


def test_seqread_unexpected_error(tmp_path, monkeypatch):
    def _raise_unexpected(*args, **kwargs):
        raise PermissionError("simulated permission failure")

    monkeypatch.setattr(os, "listdir", _raise_unexpected)
    assert imr.seqread(tmp_path) is None
