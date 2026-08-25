"""
Tests for PyDoug's io.reader module.
"""

from pathlib import Path

import numpy as np

from pydoug_core.io import imreader as imr


def test_hdf_expand_tree_returns(sample_single_hdf, sample_empty_hdf) -> None:

    result1 = imr.hdf_expand_tree(sample_single_hdf)
    result2 = imr.hdf_expand_tree(sample_empty_hdf)
    assert isinstance(result1, tuple)
    assert isinstance(result1[0], list)
    assert isinstance(result1[1], bool)
    assert isinstance(result2, tuple)
    assert not result2[1]
    assert isinstance(result2[1], bool)


def test_hdf_expand_tree_list_lengths(sample_single_hdf, sample_double_hdf) -> None:

    result1 = imr.hdf_expand_tree(sample_single_hdf)
    result2 = imr.hdf_expand_tree(sample_double_hdf)
    result3 = imr.hdf_expand_tree(sample_double_hdf, break_on_dset=True)
    assert isinstance(result1[0], list)
    assert isinstance(result2[0], list)
    assert isinstance(result3[0], list)
    assert len(result1[0]) == 1
    assert len(result2[0]) == 2
    assert len(result3[0]) == 1


def test_hdf_largest_dataset_returns(sample_single_hdf, sample_empty_hdf) -> None:

    result1 = imr.hdf_largest_dataset(sample_single_hdf)
    result2 = imr.hdf_largest_dataset(sample_empty_hdf)
    assert isinstance(result1, str)
    assert not result2


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


def test_imread_unsupported_format(sample_unsupported_format) -> None:

    assert not imr.imread(sample_unsupported_format)
