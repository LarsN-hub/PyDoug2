"""
Reading image files.
"""

from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

__all__ = ["hdf_expand_tree", "imread"]


def hdf_expand_tree(
    path: None | Path | str = None,
    *,
    hdfile: h5py.File | None = None,
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
            tab_count: int = 0
        else:
            tab_count: int = hdfname.count("/")
        for item in contents:
            name: str = str(hdfile[item].name)
            label: str = name[name.rfind("/") + 1 :]
            current_item = hdfile[item]
            if isinstance(current_item, h5py.Group):
                if printtree:
                    print((tab_count * " ") + "- " + label)
                treelist, dsetfound = hdf_expand_tree(
                    hdfile=h5py.File(hdfile[item]),
                    treelist=treelist,
                    break_on_dset=break_on_dset,
                    dsetfound=dsetfound,
                    printtree=printtree,
                )
            elif isinstance(current_item, h5py.Dataset):
                dsetfound = True
                if len(current_item.shape):
                    label += " [shape: " + str(current_item.shape) + "]"
                    if isinstance(treelist, list):
                        treelist.append(str(current_item.name))
                if printtree:
                    print((tab_count * "  ") + "- " + label)
            if dsetfound and break_on_dset:
                return treelist, dsetfound
    if isinstance(treelist, list) and len(treelist) == 0:
        return None, dsetfound
    else:
        return treelist, dsetfound


def hdf_largest_dataset(hdfile: h5py.File | Path | str) -> None | str:
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
    treelist, _ = hdf_expand_tree(hdfile=hdfile)
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
    else:
        path = Path(path)
        if path.suffix in [".h5", ".hdf", ".hdf5", ".he5"]:
            try:
                hdfile: h5py.File = h5py.File(path)
                imarray: np.ndarray = np.array(hdfile[hdf_largest_dataset(hdfile)])
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
                imarray: np.ndarray = iio.imread(path)
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
