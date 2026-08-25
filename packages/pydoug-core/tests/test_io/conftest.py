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

from pydoug_core import util


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
def sample_group_hdf(tmp_path, sample_2dgray_array) -> Path:
    """
    Fixture to create a sample hdf file with a dataset in a group.
    Returns the path.
    """

    path: Path = tmp_path / "group.hdf"
    hdfile: h5py.File = h5py.File(path, "w")
    hdfgroup: h5py.Group = hdfile.create_group("test_group1")
    _ = hdfgroup.create_dataset("test_dset1", data=sample_2dgray_array)
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
def sample_2dgray_seq(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to write a sample 3D grayscale image to a sequence of temp 2D tiff files.
    Returns the path.
    """

    path: Path = tmp_path / "3dgray"
    os.mkdir(path)
    for sliceindex in range(0, sample_3dgray_array.shape[0]):
        iio.imwrite(path / f"{sliceindex:02d}.tiff", sample_3dgray_array[sliceindex])
    with open(path / "unsupported.txt", "w"):
        pass
    return path


@pytest.fixture
def sample_2drgb_seq(tmp_path, sample_3drgb_array) -> Path:
    """
    Fixture to write a sample 3D RGB image to a sequence of temp 2D tiff files.
    Returns the path.
    """

    path: Path = tmp_path / "3drgb"
    os.mkdir(path)
    for sliceindex in range(0, sample_3drgb_array.shape[0]):
        iio.imwrite(path / f"{sliceindex:02d}.tiff", sample_3drgb_array[sliceindex])
    return path


@pytest.fixture
def sample_2d3dgrayrgb_seq(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to write a sample 3D grayscale image to a sequence of temp 2D/3D
    grayscale/RGB tiff/hdf files.
    Returns the path.
    """

    path: Path = tmp_path / "2D3Dgrayrgb"
    os.mkdir(path)
    iio.imwrite(path / "01.tiff", sample_3dgray_array[0])
    hdfile: h5py.File = h5py.File(path / "02.hdf", "w")
    _ = hdfile.create_dataset("test_dset", data=util.gray2rgb(sample_3dgray_array[1:3]))
    hdfile.close()
    iio.imwrite(path / "03.tiff", sample_3dgray_array[3:5])
    return path


@pytest.fixture
def sample_empty_dir(tmp_path) -> Path:
    """
    Fixture to create an empty folder.
    Returns the path.
    """

    path: Path = tmp_path / "empty"
    os.mkdir(path)
    return path


@pytest.fixture
def sample_misshaped_seq(tmp_path, sample_3dgray_array) -> Path:
    """
    Fixture to write a 3D grayscale image to a sequence of misshaped 2D tiff files.
    Returns the path.
    """

    path: Path = tmp_path / "misshaped"
    os.mkdir(path)
    for sliceindex in range(0, sample_3dgray_array.shape[0]):
        if sliceindex == 2:
            iio.imwrite(
                path / f"{sliceindex:02d}.tiff", sample_3dgray_array[sliceindex, 1:-1]
            )
        else:
            iio.imwrite(
                path / f"{sliceindex:02d}.tiff", sample_3dgray_array[sliceindex]
            )
    with open(path / "unsupported.txt", "w"):
        pass
    return path


@pytest.fixture
def sample_zero_size_seq(tmp_path) -> Path:
    """
    Fixture to create an image sequence with zero size arrays.
    Returns the path.
    """

    path: Path = tmp_path / "zero_size"
    os.mkdir(path)
    tifffile.imwrite(path / "01.tiff", np.array([]))
    tifffile.imwrite(path / "02.tiff", np.array([]))
    tifffile.imwrite(path / "03.tiff", np.array([]))
    return path
