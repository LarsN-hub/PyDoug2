"""
Tests for napari plugin _reader module.
"""

import os
from collections.abc import Callable
from pathlib import Path

import h5py
import imageio.v3 as iio
import pytest
from numpy import ndarray

from pydoug_napari import _reader


@pytest.fixture
def sample_3dgray_tiff(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to creat a temporary multipage tiff file for importing.
    Returns the path.
    """

    path: Path = tmp_path / "3dgray.tiff"
    iio.imwrite(path, sample_3dgray_array)
    return path


@pytest.fixture
def sample_3dgray_hdf5(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to creat a temporary hdf5 file for importing.
    Returns the path.
    """

    path: Path = tmp_path / "3dgray.hdf5"
    hdfile: h5py.File = h5py.File(path, "w")
    _ = hdfile.create_dataset("test_dset", data=sample_3dgray_array)
    hdfile.close()
    return path


@pytest.fixture
def sample_3dgray_seq(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to creat a temporary tiff sequence for importing.
    Returns the folder path.
    """

    path: Path = tmp_path / "3dgray_seq"
    os.mkdir(path)
    for index in range(0, sample_3dgray_array.shape[0]):
        iio.imwrite(path / f"3dgray_{index:02d}.tiff", sample_3dgray_array[index])
    with open(path / "unsupported.txt", "w"):
        pass
    return path


@pytest.fixture
def sample_unsupported(tmp_path) -> Path:
    """
    Fixture to creat a temporary txt file for importing.
    Returns the path.
    """

    path: Path = tmp_path / "unsupported.txt"
    with open(path, "w"):
        pass
    return path


def test_napari_get_reader_returns(
    sample_3dgray_tiff, sample_3dgray_hdf5, sample_3dgray_seq, sample_unsupported
) -> None:

    reader1: Callable | None = _reader.napari_get_reader(sample_3dgray_tiff)
    reader2: Callable | None = _reader.napari_get_reader(sample_3dgray_hdf5)
    reader3: Callable | None = _reader.napari_get_reader(sample_3dgray_seq)
    reader4: Callable | None = _reader.napari_get_reader(sample_unsupported)
    reader5: Callable | None = _reader.napari_get_reader(
        [sample_3dgray_tiff, sample_3dgray_hdf5]
    )
    assert isinstance(reader1, Callable)
    assert isinstance(reader2, Callable)
    assert isinstance(reader3, Callable)
    assert reader4 is None
    assert isinstance(reader5, Callable)


def test_reader_function(
    sample_3dgray_tiff, sample_3dgray_hdf5, sample_3dgray_seq
) -> None:

    layerdata1: list[tuple] = _reader._reader_function(sample_3dgray_tiff)
    layerdata2: list[tuple] = _reader._reader_function(sample_3dgray_hdf5)
    layerdata3: list[tuple] = _reader._reader_function(sample_3dgray_seq)
    layerdata4: list[tuple] = _reader._reader_function(
        [sample_3dgray_tiff, sample_3dgray_hdf5]
    )
    assert isinstance(layerdata1, list)
    assert isinstance(layerdata2, list)
    assert isinstance(layerdata3, list)
    assert isinstance(layerdata4, list)
    assert len(layerdata1) == 1
    assert len(layerdata2) == 1
    assert len(layerdata3) == 1
    assert len(layerdata4) == 2
    assert isinstance(layerdata1[0][0], ndarray)
    assert isinstance(layerdata1[0][1], dict)
    assert isinstance(layerdata1[0][2], str)
    assert layerdata1[0][0].shape == (5, 10, 10)
    assert layerdata1[0][1]["name"] == "3dgray"
    assert layerdata1[0][2] == "image"
