"""
Tests for Pydoug io.imwriter module.
"""

import os
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

from pydoug_core.io import imwriter as imw


def test_imwrite_2d3d_grayrgb_combos(tmp_path, sample_arrays, sample_masks) -> None:

    imw.imwrite(sample_arrays["2dgray"], tmp_path, "2dgraywrite", "tiff")
    imw.imwrite(sample_arrays["3dgray"], tmp_path, "3dgraywrite", "tiff")
    imw.imwrite(sample_arrays["2drgb"], tmp_path, "2drgbwrite", "tiff")
    imw.imwrite(sample_arrays["3drgb"], tmp_path, "3drgbwrite", "tiff")
    imw.imwrite(sample_masks["2dgray"], tmp_path, "2dboolwrite", "tiff")
    assert np.all(iio.imread(tmp_path / "2dgraywrite.tiff") == sample_arrays["2dgray"])
    assert np.all(iio.imread(tmp_path / "3dgraywrite.tiff") == sample_arrays["3dgray"])
    assert np.all(iio.imread(tmp_path / "2drgbwrite.tiff") == sample_arrays["2drgb"])
    assert np.all(iio.imread(tmp_path / "3drgbwrite.tiff") == sample_arrays["3drgb"])


def test_imwrite_extensions(tmp_path, sample_2dgray_array) -> None:

    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "tiff")
    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "tif")
    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "png")
    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "h5")
    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "hdf")
    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "hdf5")
    imw.imwrite(sample_2dgray_array, tmp_path, "2dgrayext", "he5")
    assert np.all(iio.imread(tmp_path / "2dgrayext.tiff") == sample_2dgray_array)
    assert np.all(iio.imread(tmp_path / "2dgrayext.tif") == sample_2dgray_array)
    assert np.all(iio.imread(tmp_path / "2dgrayext.png") == sample_2dgray_array)
    h5file: h5py.File = h5py.File(tmp_path / "2dgrayext.h5", "r")
    hdfile: h5py.File = h5py.File(tmp_path / "2dgrayext.hdf", "r")
    hdf5file: h5py.File = h5py.File(tmp_path / "2dgrayext.hdf5", "r")
    he5file: h5py.File = h5py.File(tmp_path / "2dgrayext.he5", "r")
    assert np.all(h5file["data"] == sample_2dgray_array)
    assert np.all(hdfile["data"] == sample_2dgray_array)
    assert np.all(hdf5file["data"] == sample_2dgray_array)
    assert np.all(he5file["data"] == sample_2dgray_array)


def test_imwrite_file_exists_error(tmp_path, sample_2dgray_array) -> None:

    with pytest.raises(FileExistsError):
        iio.imwrite(tmp_path / "exists.tiff", sample_2dgray_array)
        imw._write(sample_2dgray_array, tmp_path / "exists.tiff", "tiff")


def test_imwrite_full_path_handling(tmp_path, sample_2dgray_array) -> None:

    imw.imwrite(sample_2dgray_array, tmp_path, "fullpath.tiff", "tiff")
    assert np.all(iio.imread(tmp_path / "fullpath.tiff") == sample_2dgray_array)


def test_imwrite_exception_handling(tmp_path, monkeypatch, sample_2dgray_array) -> None:
    def _raise_unexpected(*args, **kwargs):
        raise ValueError("Simulated value error.")

    iio.imwrite(tmp_path / "exceptions.tiff", sample_2dgray_array)
    imw.imwrite(sample_2dgray_array, tmp_path, "exceptions", "tiff")
    imw.imwrite(sample_2dgray_array, tmp_path / "nonexistant", "exceptions", "tiff")
    assert not Path(tmp_path / "nonexistant/exceptions.tiff").exists()
    monkeypatch.setattr(imw, "_write", _raise_unexpected)
    imw.imwrite(sample_2dgray_array, tmp_path, "exceptions2", "tiff")
    assert not Path(tmp_path / "exceptions2.tiff").exists()


def test_seqwrite_2d3d_grayrgb_combos(tmp_path, sample_arrays) -> None:

    imw.seqwrite(sample_arrays["2dgray"], tmp_path, "2dgraywrite", "tiff")
    imw.seqwrite(sample_arrays["3dgray"], tmp_path, "3dgraywrite", "tiff")
    imw.seqwrite(
        sample_arrays["3dgray"], tmp_path, "3dgraywrite_auto", "tiff", numworkers=None
    )
    imw.seqwrite(
        sample_arrays["3dgray"], tmp_path, "3dgraywrite_two", "tiff", numworkers=2
    )
    imw.seqwrite(sample_arrays["2drgb"], tmp_path, "2drgbwrite", "tiff")
    imw.seqwrite(sample_arrays["3drgb"], tmp_path, "3drgbwrite", "tiff")
    assert np.all(
        iio.imread(tmp_path / "2dgraywrite/2dgraywrite.tiff") == sample_arrays["2dgray"]
    )
    assert np.all(
        iio.imread(tmp_path / "2drgbwrite/2drgbwrite.tiff") == sample_arrays["2drgb"]
    )
    read_3dgray_array: np.ndarray = np.empty(
        sample_arrays["3dgray"].shape, sample_arrays["3dgray"].dtype
    )
    filenames_3dgray: list[str] = sorted(os.listdir(tmp_path / "3dgraywrite"))
    for i in range(0, read_3dgray_array.shape[0]):
        read_3dgray_array[i] = iio.imread(
            tmp_path / f"3dgraywrite/{filenames_3dgray[i]}"
        )
    assert np.all(read_3dgray_array == sample_arrays["3dgray"])
    read_3drgb_array: np.ndarray = np.empty(
        sample_arrays["3drgb"].shape, sample_arrays["3drgb"].dtype
    )
    filenames_3drgb: list[str] = sorted(os.listdir(tmp_path / "3drgbwrite"))
    for i in range(0, read_3drgb_array.shape[0]):
        read_3drgb_array[i] = iio.imread(tmp_path / f"3drgbwrite/{filenames_3drgb[i]}")
    assert np.all(read_3drgb_array == sample_arrays["3drgb"])
    read_3dgrayauto_array: np.ndarray = np.empty(
        sample_arrays["3dgray"].shape, sample_arrays["3dgray"].dtype
    )
    filenames_3dgrayauto: list[str] = sorted(os.listdir(tmp_path / "3dgraywrite_auto"))
    for i in range(0, read_3dgrayauto_array.shape[0]):
        read_3dgrayauto_array[i] = iio.imread(
            tmp_path / f"3dgraywrite_auto/{filenames_3dgrayauto[i]}"
        )
    assert np.all(read_3dgrayauto_array == sample_arrays["3dgray"])
    read_3dgraytwo_array: np.ndarray = np.empty(
        sample_arrays["3dgray"].shape, sample_arrays["3dgray"].dtype
    )
    filenames_3dgraytwo: list[str] = sorted(os.listdir(tmp_path / "3dgraywrite_two"))
    for i in range(0, read_3dgraytwo_array.shape[0]):
        read_3dgraytwo_array[i] = iio.imread(
            tmp_path / f"3dgraywrite_two/{filenames_3dgraytwo[i]}"
        )
    assert np.all(read_3dgraytwo_array == sample_arrays["3dgray"])


def test_seqwrite_extensions(tmp_path, sample_3dgray_array) -> None:

    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_tiff_write", "tiff")
    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_tif_write", "tif")
    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_png_write", "png")
    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_h5_write", "h5")
    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_hdf_write", "hdf")
    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_hdf5_write", "hdf5")
    imw.seqwrite(sample_3dgray_array, tmp_path, "3dgray_he5_write", "he5")
    read_3dgray_tiff: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_tiff: list[str] = sorted(
        os.listdir(tmp_path / "3dgray_tiff_write")
    )
    for i in range(0, read_3dgray_tiff.shape[0]):
        read_3dgray_tiff[i] = iio.imread(
            tmp_path / f"3dgray_tiff_write/{filenames_3dgray_tiff[i]}"
        )
    assert np.all(read_3dgray_tiff == sample_3dgray_array)
    read_3dgray_tif: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_tif: list[str] = sorted(os.listdir(tmp_path / "3dgray_tif_write"))
    for i in range(0, read_3dgray_tif.shape[0]):
        read_3dgray_tif[i] = iio.imread(
            tmp_path / f"3dgray_tif_write/{filenames_3dgray_tif[i]}"
        )
    assert np.all(read_3dgray_tif == sample_3dgray_array)
    read_3dgray_png: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_png: list[str] = sorted(os.listdir(tmp_path / "3dgray_png_write"))
    for i in range(0, read_3dgray_png.shape[0]):
        read_3dgray_png[i] = iio.imread(
            tmp_path / f"3dgray_png_write/{filenames_3dgray_png[i]}"
        )
    assert np.all(read_3dgray_png == sample_3dgray_array)
    read_3dgray_h5: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_h5: list[str] = sorted(os.listdir(tmp_path / "3dgray_h5_write"))
    for i in range(0, read_3dgray_h5.shape[0]):
        current_hdf: h5py.File = h5py.File(
            tmp_path / f"3dgray_h5_write/{filenames_3dgray_h5[i]}", "r"
        )
        read_3dgray_h5[i] = current_hdf["data"]
        current_hdf.close()
    assert np.all(read_3dgray_h5 == sample_3dgray_array)
    read_3dgray_hdf: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_hdf: list[str] = sorted(os.listdir(tmp_path / "3dgray_hdf_write"))
    for i in range(0, read_3dgray_hdf.shape[0]):
        current_hdf: h5py.File = h5py.File(
            tmp_path / f"3dgray_hdf_write/{filenames_3dgray_hdf[i]}", "r"
        )
        read_3dgray_hdf[i] = current_hdf["data"]
        current_hdf.close()
    assert np.all(read_3dgray_hdf == sample_3dgray_array)
    read_3dgray_hdf5: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_hdf5: list[str] = sorted(
        os.listdir(tmp_path / "3dgray_hdf5_write")
    )
    for i in range(0, read_3dgray_hdf5.shape[0]):
        current_hdf: h5py.File = h5py.File(
            tmp_path / f"3dgray_hdf5_write/{filenames_3dgray_hdf5[i]}", "r"
        )
        read_3dgray_hdf5[i] = current_hdf["data"]
        current_hdf.close()
    assert np.all(read_3dgray_hdf5 == sample_3dgray_array)
    read_3dgray_he5: np.ndarray = np.empty(
        sample_3dgray_array.shape, sample_3dgray_array.dtype
    )
    filenames_3dgray_he5: list[str] = sorted(os.listdir(tmp_path / "3dgray_he5_write"))
    for i in range(0, read_3dgray_he5.shape[0]):
        current_hdf: h5py.File = h5py.File(
            tmp_path / f"3dgray_he5_write/{filenames_3dgray_he5[i]}", "r"
        )
        read_3dgray_he5[i] = current_hdf["data"]
        current_hdf.close()
    assert np.all(read_3dgray_he5 == sample_3dgray_array)


def test_seqwrite_exception_handling(
    tmp_path, monkeypatch, sample_3dgray_array
) -> None:
    def _raise_unexpected(*args, **kwargs):
        raise ValueError("Simulated value error.")

    os.mkdir(tmp_path / "already_exists")
    imw.seqwrite(sample_3dgray_array, tmp_path, "already_exists")
    assert len(sorted(os.listdir(tmp_path / "already_exists"))) == 0
    imw.seqwrite(sample_3dgray_array, tmp_path / "does_not_exist", "test")
    assert not Path(tmp_path / "does_not_exist").exists()
    monkeypatch.setattr(imw, "imwrite", _raise_unexpected)
    imw.seqwrite(sample_3dgray_array, tmp_path, "test1", numworkers=2)
    assert len(sorted(os.listdir(tmp_path / "test1"))) == 0
    monkeypatch.setattr(os, "mkdir", _raise_unexpected)
    imw.seqwrite(sample_3dgray_array, tmp_path, "test2")
    assert not Path(tmp_path / "test2").exists()
