"""
Add/find morphological features in images.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation
from pydoug_core.util import axstr2int, is3d, isrgb, min_int_dtype

__all__ = ["euclidian_dt", "mis", "chords"]


@register_operation(category="features", display_name="Distance")
def euclidian_dt(
    imarray: np.ndarray,
    Pixel_Size: Annotated[float, {"step": 0.0001, "min": 0}] = 1,
    Round_Values: bool = False,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Transform a binary image into the distance domain by calculating the Euclidian
    distance from each foreground point to the nearest background point. Image must be
    2D/3D binary. Convenience wrapper around scipy's ndimage.distance_transform_edt:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Pixel_Size: float
        The real-world length of a single pixel edge.
    Round_Values: bool
        Whether to round returned values to the nearest integer (True) or leave as exact
        floating point distance (False). Default is False.
    Perform_Slicewise: bool
        Whether to perform the operation slicewise for 3D images (True) or in bulk
        (False). Default is False.

    Returns
    -------
    edtarray: np.ndarray
        The image array transformed into the distance domain.
    """

    if isrgb(imarray):
        raise ValueError("Image cannot be RGB!")

    from scipy.ndimage import distance_transform_edt as edt

    if Perform_Slicewise and is3d(imarray):
        edtarray: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            edtarray[index] = edt(imarray[index], Pixel_Size)

        return (
            np.round(edtarray).astype(min_int_dtype(edtarray))
            if Round_Values
            else edtarray
        )

    edtarray = edt(imarray, Pixel_Size)  # type: ignore

    return (
        np.round(edtarray).astype(min_int_dtype(edtarray)) if Round_Values else edtarray
    )


@register_operation(category="features", display_name="Maximum Inscribed Spheres")
def mis(
    imarray: np.ndarray,
    Method: Literal[
        "Brute Force", "Convolution", "Distance Transform", "ImageJ"
    ] = "Distance Transform",
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Smooth: bool = True,
    Return_Diameters: bool = False,
    Round_Values: bool = False,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Find the maximum inscribed spheres for a binary image. Image must be 2D/3D binary.
    Convenience wrapper around porespy's filters.local_thickness:
    https://porespy.org/autoapi/porespy/filters/local_thickness.html#

    Parameters
    ----------
    imarray: np.ndarray
        The image to find maximum inscribed spheres in.
    Method: Literal
        The method for finding the inscribed spheres. Default is "Distance Transform".
        - "Brute Force": Brute force algorithm.
        - "Distance Transform": Uses distance transforms to dilate/erode each radius.
        - "ImageJ": Uses ImageJ's algorithm (faster "Brute Force", less accurate).
        - "Convolution": FFT-based convolution to dilate/erode each radius.
    Pixel_Size: float
        The edge length of pixels in real-world units.
    Smooth: bool
        Whether to remove small facets at borders (True) or not (False). Default True.
    Return_Diameters: bool
        Whether to return the diameter of the spheres (True) or the radii (False).
        Default is False.
    Round_Values: bool
        Whether to round returned values to the nearest integer (True) or leave as exact
        floating point distance (False). Default is False.
    Perform_Slicewise: bool
        Whether or not to perform the operation slicewise. Default is False.

    Returns
    -------
    mis_array: np.ndarray
        The maximum inscribed spheres image.
    """

    if isrgb(imarray):
        raise ValueError("Image cannot be RGB!")

    from porespy.filters import local_thickness

    def _mis(imarray) -> np.ndarray:

        if Method == "Brute Force":
            method: str = "bf"
        elif Method == "Distance Transform":
            method = "dt"
        elif Method == "ImageJ":
            method = "imj"
        else:
            method = "conv"

        rad_array: np.ndarray = (
            local_thickness(imarray.astype(np.bool), method=method, smooth=Smooth)
            * Pixel_Size
        )

        if Return_Diameters:
            return rad_array * 2
        return rad_array

    if Perform_Slicewise and is3d(imarray):
        mis_array: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            mis_array[index] = _mis(imarray[index])

        return (
            np.round(mis_array).astype(min_int_dtype(mis_array))
            if Round_Values
            else mis_array
        )

    mis_array: np.ndarray = _mis(imarray)
    return (
        np.round(mis_array).astype(min_int_dtype(mis_array))
        if Round_Values
        else mis_array
    )


@register_operation(category="features", display_name="Chords")
def chords(
    imarray: np.ndarray,
    Axis: Literal["X", "Y", "Z"] = "Y",
    Spacing: Annotated[int, {"min": 1}] = 1,
) -> np.ndarray:
    """
    Return an image with the segmented phase of a binary image replaced by random
    chords. This can be used to create a chord length distribution in the "Measure" tab.
    All non-zero values in the input image will be treated as the segmented phase. Image
    must be 2D/3D binary. Convenience wrapper around porespy's filters.apply_chords:
    https://porespy.org/autoapi/porespy/filters/apply_chords.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to add chords to.
    Axis: Literal
        The axis along which to apply chords. Will return an error if "Z" is selected
        and the image is not 3D.
    Spacing: int
        The spacing between chords (in pixels). Minimum is 1.

    Returns
    -------
    chord_array: np.ndarray
        The image with chords added.
    """

    if isrgb(imarray):
        raise ValueError("Image cannot be RGB!")

    if Axis == "Z" and not is3d(imarray):
        raise ValueError("'Z' axis selected but image is not 3D!")

    from porespy.filters import apply_chords

    return apply_chords(imarray != 0, Spacing, axstr2int(imarray, Axis), False, False)


@register_operation(category="features", display_name="Path Length")
def path_length(
    imarray: np.ndarray,
    Axis: Literal["X", "Y", "Z"] = "Y",
    Connectivity: Literal[1, 2, 3] = 2,
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Estimate_Boundaries: bool = False,
    Round_Values: bool = False,
) -> np.ndarray:
    """
    Find the minimum path length from one end of the chosen axis to the other through
    each pixel. This can be used to create a path length distribution by plotting the
    histogram of the output image. Convenience wrapper around scikit-image's
    graph.pixel_graph and scipy's sparse.csgraph.dijkstra:
    https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.pixel_graph
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html#scipy.sparse.csgraph.dijkstra

    Parameters
    ----------
    imarray: np.ndarray
        The segmented image.
    Axis: Literal
        The axis along which to find the path length. Raises an error if "Z" is selected
        and the image is not 3D.
    Connectivity: Literal[1, 2, 3]
        The method to use for determing segment connectivity. Default is 2.
        - 1: Only pixels with faces in contact are considered connected.
            - In 2D, this means only four most adjacent pixels.
            - In 3D, this means only the 6 most adjacent voxels.
        - 2: Includes 1, and adds pixels with edges in contact.
            - In 2D, this means the surrounding 8 pixels.
            - In 3D, this means only the 18 most adjacent voxels.
        - 3: (3D only) Includes 1-2, and adds pixels with corners in contact.
            - In 3D, this means the surrounding 26 voxels.
    Pixel_Size: float
        The edge-length of pixels in real-world units.
    Estimate_Boundaries: bool
        Whether to estimate the boundaries of the segmented region along the axis (True)
        or to just use the axis borders (False). If False and no connected segmented
        pixels contact both axis boundaries, no path lengths will be measured. If True,
        assumes the nearest and furthest segmented pixel in each column are boundaries.
    Round_Values: bool
        Whether to round returned values to the nearest integer (True) or leave as exact
        floating point distance (False). Default is False.

    Returns
    -------
    plarray: np.ndarray
        The image with each pixel's value reassigned as its path length.
    """

    if isrgb(imarray):
        raise ValueError("Image cannot be RGB!")

    if Axis == "Z" and not is3d(imarray):
        raise ValueError("'Z' axis selected but image is not 3D!")

    if Connectivity == 3 and not is3d(imarray):
        Connectivity = 2

    from scipy.sparse.csgraph import dijkstra
    from skimage.graph import pixel_graph

    eval_array: np.ndarray = imarray != 0
    graph, nodes = pixel_graph(
        eval_array, connectivity=Connectivity, sparse_type="array"
    )
    axint: int = axstr2int(imarray, Axis)

    if Estimate_Boundaries:
        top_array: np.ndarray = np.zeros(imarray.shape, np.bool)
        bot_array: np.ndarray = np.copy(top_array)
        top_indices: np.ndarray = np.argmax(eval_array, axis=axint)
        bot_indices: np.ndarray = (
            imarray.shape[axint]
            - 1
            - np.argmax(
                np.flip(eval_array, axis=axint),
                axis=axint,
            )
        )
        check_array: np.ndarray = np.count_nonzero(eval_array, axis=axint) != 0
        top_coords: list[np.ndarray] = []
        bot_coords: list[np.ndarray] = []
        for ax_index in range(0, imarray.ndim):
            cur_top: np.ndarray = (
                top_indices
                if ax_index == axint
                else np.arange(0, imarray.shape[ax_index])
            )
            cur_top = np.delete(cur_top, np.argwhere(np.logical_not(check_array)))
            top_coords.append(cur_top)
            cur_bot: np.ndarray = (
                bot_indices
                if ax_index == axint
                else np.arange(0, imarray.shape[ax_index])
            )
            cur_bot = np.delete(cur_bot, np.argwhere(np.logical_not(check_array)))
            bot_coords.append(cur_bot)
        top_array[tuple(top_coords)] = 1
        bot_array[tuple(bot_coords)] = 1

    else:
        top_array: np.ndarray = np.copy(eval_array)
        bot_array: np.ndarray = np.copy(eval_array)
        if axint == 0:
            top_array[1:] = 0
            bot_array[:-1] = 0
        elif axint == 1:
            top_array[:, 1:] = 0
            bot_array[:, :-1] = 0
        else:
            top_array[:, :, 1:] = 0
            bot_array[:, :, :-1] = 0

    top_nodes: np.ndarray = np.squeeze(np.argwhere(np.ravel(top_array)))
    bot_nodes: np.ndarray = np.squeeze(np.argwhere(np.ravel(bot_array)))
    top_indices: np.ndarray = np.searchsorted(nodes, top_nodes)
    bot_indices: np.ndarray = np.searchsorted(nodes, bot_nodes)
    top_dist = dijkstra(graph, indices=top_indices, min_only=True)
    bot_dist = dijkstra(graph, indices=bot_indices, min_only=True)
    tot_dist: np.ndarray = (top_dist + bot_dist) * Pixel_Size
    tot_dist[tot_dist == np.inf] = 0
    plarray: np.ndarray = np.zeros(imarray.shape)
    plarray[np.unravel_index(nodes, imarray.shape)] = tot_dist

    return np.round(plarray).astype(min_int_dtype(plarray)) if Round_Values else plarray
