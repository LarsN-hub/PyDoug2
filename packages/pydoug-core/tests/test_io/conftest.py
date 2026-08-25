"""
Fixtures for core io tests.
"""

import os
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest
import tifffile


@pytest.fixture(params=["2dgray", "3dgray", "2drgb", "3drgb"])
def sample_tiff(request, tmp_path, sample_arrays) -> tuple[Path, str]:
    """
    Fixture to write array matching request.param to temp tiff file.
    Returns tuple with path and kind-name string.
    """

    kind: str = request.param
    array: np.ndarray = sample_arrays[kind]
    path: Path = tmp_path / f"{kind}.tiff"
    iio.imwrite(path, array)
    return path, kind


@pytest.fixture(params=["2dgray", "3dgray", "2drgb", "3drgb"])
def sample_hdf(request, tmp_path, sample_arrays) -> tuple[Path, str]:
    """
    Fixture to write array matching request.param to temp hdf file.
    Returns tuple with path and kind-name string.
    """

    kind: str = request.param
    array: np.ndarray = sample_arrays[kind]
    path: Path = tmp_path / f"{kind}.hdf"
    hdfile: h5py.File = h5py.File(path, "w")
    _ = hdfile.create_dataset("test_dset", data=array)
    hdfile.close()
    return path, kind


@pytest.fixture
def sample_single_hdf(tmp_path, sample_2dgray_array) -> Path:
    """
    Fixture to create a sample hdf file with one dataset.
    Returns the path.
    """

    path: Path = tmp_path / "single.hdf"
    hdfile: h5py.File = h5py.File(path, "w")
    _ = hdfile.create_dataset("test_dset", data=sample_2dgray_array)
    hdfile.close()
    return path


@pytest.fixture
def sample_double_hdf(tmp_path, sample_2dgray_array) -> Path:
    """
    Fixture to create a sample hdf file with two datasets.
    Returns the path.
    """

    path: Path = tmp_path / "double.hdf"
    hdfile: h5py.File = h5py.File(path, "w")
    _ = hdfile.create_dataset("test_dset1", data=sample_2dgray_array)
    _ = hdfile.create_dataset("test_dset2", data=sample_2dgray_array)
    hdfile.close()
    return path


@pytest.fixture
def sample_empty_tiff(tmp_path) -> Path:
    """
    Fixture to create a sample empty tiff file.
    Returns the path.
    """

    path: Path = tmp_path / "empty.tiff"
    tifffile.imwrite(path, np.array([]))
    return path


@pytest.fixture
def sample_empty_hdf(tmp_path) -> Path:
    """
    Fixture to create a sample empty hdf file.
    Returns the path.
    """

    path: Path = tmp_path / "empty.hdf"
    hdfile: h5py.File = h5py.File(path, "w")
    hdfile.close()
    return path


@pytest.fixture
def sample_unsupported_format(tmp_path) -> Path:
    """
    Fixture to create a file with an unsupported format.
    Returns the path.
    """

    path: Path = tmp_path / "unsupported.txt"
    with open(path, "w"):
        pass
    return path


@pytest.fixture
def sample_3dgray_seq(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to write a sample 3D grayscale image to a temp tiff sequence.
    Returns the path.
    """

    path: Path = tmp_path / "3dgray"
    os.mkdir(path)
    for sliceindex in range(0, sample_3dgray_array.shape[0]):
        iio.imwrite(path / f"{sliceindex:02d}.tiff", sample_3dgray_array[sliceindex])
    return path


@pytest.fixture
def sample_3drgb_seq(tmp_path, sample_3drgb_array) -> Path:
    """
    Fixture to write a sample 3D RGB image to a temp tiff sequence.
    Returns the path.
    """

    path: Path = tmp_path / "3drgb"
    os.mkdir(path)
    for sliceindex in range(0, sample_3drgb_array.shape[0]):
        iio.imwrite(path / f"{sliceindex:02d}.tiff", sample_3drgb_array[sliceindex])
    return path
