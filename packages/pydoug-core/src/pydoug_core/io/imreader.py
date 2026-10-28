"""
Reading image files.
"""

from pathlib import Path

import h5py
import numpy as np

__all__ = ["imread", "seqread"]


def _hdf_expand_tree(
    path: None | Path | str = None,
    *,
    hdfile: h5py.File | h5py.Group | None = None,
    treelist: list[str] | None = None,
    break_on_dset: bool = False,
    dsetfound: bool = False,
    printtree: bool = False,
) -> tuple[list | None, bool]:
    """
    Recursive function to expand the file tree of an hdf file.

    Parameters
    ----------
    path: None | Path | str
        Path to file. This is optional and ignored only if hdfile is provided.
    hdfile: h5py.File | None (optional)
        An hdf file object created from h5py.File.
        This is optional if path is provided.
    treelist: list[str] | None (optional)
        The current list of directories in the hdf file tree.
        This is optional if path is provided.
    break_on_dset: bool
        Whether or not to break the recursion upon finding the first dataset.
    dsetfound: bool
        Indicates if a dataset was found during recursive expansion.
    printtree: bool
        Whether or not to print the file tree.

    Returns
    -------
    treelist: list[str] | None
        A list containing the hdf file's directory tree.
        Returns None if no datasets are found.
    dsetfound: bool
        Indicates if a dataset was found during recursive expansion.
    """

    if not treelist:
        treelist = []
    if path:
        path = str(path)
    if not break_on_dset or (break_on_dset and not dsetfound):
        if not hdfile:
            hdfile = h5py.File(path)
        contents: list[str] = list(hdfile.keys())
        hdfname: str = str(hdfile.name)
        if hdfname == "/":
            tabcount: int = 0
        else:
            tabcount: int = hdfname.count("/")
        for item in contents:
            name: str = str(hdfile[item].name)
            label: str = name[name.rfind("/") + 1 :]
            currentitem = hdfile[item]
            if isinstance(currentitem, h5py.Group):
                if printtree:
                    print((tabcount * " ") + "- " + label)
                treelist, dsetfound = _hdf_expand_tree(
                    hdfile=currentitem,
                    treelist=treelist,
                    break_on_dset=break_on_dset,
                    dsetfound=dsetfound,
                    printtree=printtree,
                )
            elif isinstance(currentitem, h5py.Dataset):
                dsetfound = True
                if len(currentitem.shape):
                    label += " [shape: " + str(currentitem.shape) + "]"
                    if isinstance(treelist, list):
                        treelist.append(str(currentitem.name))
                if printtree:
                    print((tabcount * "  ") + "- " + label)
            if dsetfound and break_on_dset:
                return treelist, dsetfound
    if isinstance(treelist, list) and len(treelist) == 0:
        return None, dsetfound
    else:
        return treelist, dsetfound


def _hdf_largest_dataset(hdfile: h5py.File | Path | str) -> None | str:
    """
    Automatically locate the largest dataset within an hdf file tree.

    Parameters
    ----------
    file: h5py.File | Path | str
        Path to file or an h5py.File object.

    Returns
    -------
    largestpath: None | str
        The path within the hdf file to the largest item.
        Returns None if no dataset is found.
    """

    if not isinstance(hdfile, h5py.File):
        hdfile = h5py.File(hdfile)
    treelist, _ = _hdf_expand_tree(hdfile=hdfile)
    prevbytes: int = 0
    currentlargest = None
    if isinstance(treelist, list):
        for directory in treelist:
            current_directory = hdfile[directory]
            if isinstance(current_directory, h5py.Dataset):
                if current_directory.nbytes > prevbytes:
                    currentlargest = directory
                    prevbytes = current_directory.nbytes
    return currentlargest


def imread(path: None | Path | str) -> None | np.ndarray:
    """
    Read a single image file into a numpy array.

    Parameters
    ----------
    path: None | Path | str
        Path to the image file.

    Returns
    -------
    None | np.ndarray
        The image data as a numpy array, or None if the path is not provided.
    """

    if not path:
        return None

    from imageio.v3 import imread as iioread

    path = Path(path)
    if path.suffix in [".h5", ".hdf", ".hdf5", ".he5"]:
        try:
            hdfile: h5py.File = h5py.File(path)
            imarray: np.ndarray = np.array(hdfile[_hdf_largest_dataset(hdfile)])
            hdfile.close()
            return imarray
        except FileNotFoundError:
            print(f"File does not exist: {path}")
            return None
        except Exception as e:
            print(f"Error reading hdf data from {path}: {e}")
            return None
    else:
        try:
            imarray: np.ndarray = iioread(path)
            if imarray.shape == (0,):
                return None
            return imarray
        except FileNotFoundError:
            print(f"File does not exist: {path}")
            return None
        except OSError:
            print(f"Unsupported file type: {path}")
            return None
        except Exception as e:
            print(f"Error reading data from {path}: {e}")
            return None


def seqread(path: None | Path | str, numworkers: int | None = 1) -> None | np.ndarray:
    """
    Read a sequence of image files into a numpy array.

    Parameters
    ----------
    path: None | Path | str
        Path to the directory containing the image sequence.
    numworkers: int | None
        Number of threads to use for parallel file reading.
            - 1 (default): fully sequential, no threading.
            - >1: use this many worker threads.
            - None: auto-detect a safe worker count.

    Returns
    -------
    None | np.ndarray
        The image data as a numpy array, or None if the path is not provided.
    """

    if not path:
        return None

    import os
    from concurrent.futures import ThreadPoolExecutor

    from pydoug_core import util
    from pydoug_core.io.io_util import _auto_worker_count

    path = Path(path)
    try:
        filenames: list[str] = sorted(os.listdir(path))
        if not filenames:
            return None
    except FileNotFoundError:
        print(f"Directory does not exist: {path}")
        return None
    except Exception as e:
        print(f"Error reading data from {path}: {e}")
        return None

    rgb: bool = False
    xyshape: tuple[int, int] | None = None
    numslices: int = 0

    if numworkers is None or numworkers > 1:
        for name in filenames:
            firstarray: np.ndarray | None = imread(path / name)
            if firstarray is not None:
                if util.isrgb(firstarray):
                    rgb = True
                    xyshape = firstarray.shape[-3:-1]
                else:
                    rgb = False
                    xyshape = firstarray.shape[-2:]
                if numworkers is None:
                    numworkers = _auto_worker_count(firstarray, len(filenames))
                continue
        with ThreadPoolExecutor(max_workers=numworkers) as executor:
            readresults = list(
                executor.map(lambda name: imread(path / name), filenames)
            )
            arrays: list[np.ndarray] = []
            numslices: int = 0
            for array in readresults:
                if array is not None:
                    arrays.append(array)
                    numslices += array.shape[0] if util.is3d(array) else 1

    else:
        arrays: list[np.ndarray] = []
        for name in filenames:
            currentarray: np.ndarray | None = imread(path / name)
            if currentarray is None:
                continue
            if util.isrgb(currentarray):
                rgb = True
            if util.isrgb(currentarray):
                currentxy = currentarray.shape[-3:-1]
            else:
                currentxy = currentarray.shape[-2:]
            if xyshape is None:
                xyshape = currentxy
            elif currentxy != xyshape:
                print(
                    f"Inconsistent XY dimensions in sequence: {name} has {currentxy}, "
                    f"expected {xyshape}"
                )
                return None
            numslices += currentarray.shape[0] if util.is3d(currentarray) else 1
            arrays.append(currentarray)

    if not arrays or not xyshape:
        return None

    finaldtype = np.result_type(*(a.dtype for a in arrays))
    finalshape = (numslices, *xyshape, 3) if rgb else (numslices, *xyshape)
    imarray = np.empty(finalshape, dtype=finaldtype)

    offset = 0
    for currentarray in arrays:
        if rgb and not util.isrgb(currentarray):
            currentarray = util.gray2rgb(currentarray)
        if not util.is3d(currentarray):
            currentarray = np.expand_dims(currentarray, 0)
        n = currentarray.shape[0]
        imarray[offset : offset + n] = currentarray
        offset += n

    return imarray
