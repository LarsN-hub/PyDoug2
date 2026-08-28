"""
Writing image files
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

from pydoug_core import util
from pydoug_core.io import io_util

__all__ = ["imwrite", "seqwrite"]


def _write(imarray: np.ndarray, path: Path, ext: str) -> None:
    """
    Write an image array to a file.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to write.
    path: Path
        The full path (name and extension included) to write the image array.
    ext: str
        The extension to use for writing the file.
    """

    if path.exists():
        raise FileExistsError
    if ext in ["h5", "hdf", "hdf5", "he5"]:
        hdfile: h5py.File = h5py.File(path, "w")
        _ = hdfile.create_dataset("data", data=imarray)
        hdfile.close()
    else:
        iio.imwrite(path, imarray)


def imwrite(
    imarray: np.ndarray, path: Path | str, name: str, ext: str = "tiff"
) -> None:
    """
    Write an image array to a file.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to write.
    path: Path | str
        The path to write the image array.
    name: str
        The name of the file to be written.
    ext: str
        The extension to use for writing the file.
    """

    path = Path(path)
    if name[-len(ext) :] == ext:
        name = name[: -len(ext) - 1]
    path = path / f"{name}.{ext}"
    try:
        _write(imarray, path, ext)
    except FileExistsError:
        print(f"File already exists: {path}")
        return None
    except FileNotFoundError:
        print(f"Parent directory does not exist: {path.stem}")
        return None
    except Exception as e:
        print(f"Error writing data to {path}: {e}")
        return None


def seqwrite(
    imarray: np.ndarray,
    path: Path | str,
    name: str,
    ext: str = "tiff",
    *,
    numworkers: int | None = 1,
) -> None:
    """
    Write an image array to a file sequence into a new directory.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to write.
    path: Path | str
        The path to the parent directory that will hold the new directory.
    ext: str
        The extension to use for writing each file.
    """

    path = Path(path)
    path = path / f"{name}"
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory already exists: {path}")
        return None
    except FileNotFoundError:
        print(f"Parent directory does not exist: {path.stem}")
        return None
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return None

    if not util.is3d(imarray):
        imwrite(imarray, path, name, ext)
        return

    filenames: list[str] = [
        f"{name}_{index:04d}" for index in range(0, imarray.shape[0])
    ]

    def _write_slice(index: int) -> None:
        imwrite(imarray[index], path, filenames[index], ext)

    if numworkers is None:
        numworkers = io_util._auto_worker_count(imarray[0], imarray.shape[0])

    if numworkers > 1:
        try:
            with ThreadPoolExecutor(max_workers=numworkers) as executor:
                futures = {
                    executor.submit(_write_slice, index): index
                    for index in range(imarray.shape[0])
                }
            for future in as_completed(futures):
                future.result()
        except Exception as e:
            print(f"Error writing parallel sequence to {path}: {e}")
            return None

    else:
        for index in range(0, imarray.shape[0]):
            _write_slice(index)
