"""
Morphological filters.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.morphology import dilation as ski_dilation
from skimage.morphology import erosion as ski_erosion

from pydoug_core.registry import register_operation
from pydoug_core.util import is3d, isrgb

__all__ = [
    "dilation",
    "erosion",
    "closing",
    "opening",
    "black_tophat",
    "white_tophat",
    "remove_labeled_objects",
    "remove_holes",
    "convex_hull",
    "thin",
    "extrema",
]


@register_operation(category="morphology", display_name="Dilation")
def dilation(
    imarray: np.ndarray,
    Iterations: Annotated[int, {"min": 0}] = 1,
    Edge_Mode: Literal[
        "Constant", "Ignore", "Max", "Min", "Mirror", "Nearest", "Reflect", "Wrap"
    ] = "Reflect",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Dilate an image. Accepts 2D/3D grayscale/RGB images. Sets a pixel to the max of all
    local pixel values. Widens bright areas and shrinks dark areas. Convenience wrapper
    around scikit-image's morphology.dilation:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation

    Parameters
    ----------
    imarray: np.ndarray
        The image to dilate.
    Iterations: int
        The number of times to dilate the image. Default is 1.
    Edge_Mode: Literal
        How the algorithm should handle image borders. Default is "Reflect".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Ignore": Pixels outside the border do not influence the result.
        - "Max": Pixels outside the border are set to max dtype value (no influence).
        - "Min": Pixels outside the border are set to min dtype value.
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    dil_array: np.ndarray
        The dilated image.
    """

    def _dilate(imarray, Iterations, Edge_Mode, Constant_Value) -> np.ndarray:

        darray: np.ndarray = np.copy(imarray)
        for _ in range(0, Iterations):
            darray = ski_dilation(darray, mode=Edge_Mode.lower(), cval=Constant_Value)

        return darray

    if Perform_Slicewise and is3d(imarray):
        dil_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            dil_array[index] = _dilate(
                imarray[index], Iterations, Edge_Mode, Constant_Value
            )

        return dil_array

    return _dilate(imarray, Iterations, Edge_Mode, Constant_Value)


@register_operation(category="morphology", display_name="Erosion")
def erosion(
    imarray: np.ndarray,
    Iterations: Annotated[int, {"min": 0}] = 1,
    Edge_Mode: Literal[
        "Constant", "Ignore", "Max", "Min", "Mirror", "Nearest", "Reflect", "Wrap"
    ] = "Reflect",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Erode an image. Accepts 2D/3D grayscale/RGB images. Sets a pixel to the min of all
    local pixel values. Widens dark areas and shrinks bright areas. Convenience wrapper
    around scikit-image's morphology.erosion:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion

    Parameters
    ----------
    imarray: np.ndarray
        The image to erode.
    Iterations: int
        The number of times to erode the image. Default is 1.
    Edge_Mode: Literal
        How the algorithm should handle image borders. Default is "Reflect".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Ignore": Pixels outside the border do not influence the result.
        - "Max": Pixels outside the border are set to max dtype value (no influence).
        - "Min": Pixels outside the border are set to min dtype value.
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    er_array: np.ndarray
        The eroded image.
    """

    def _erode(imarray, Iterations, Edge_Mode, Constant_Value) -> np.ndarray:

        earray: np.ndarray = np.copy(imarray)
        for _ in range(0, Iterations):
            earray = ski_erosion(earray, mode=Edge_Mode.lower(), cval=Constant_Value)

        return earray

    if Perform_Slicewise and is3d(imarray):
        er_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            er_array[index] = _erode(
                imarray[index], Iterations, Edge_Mode, Constant_Value
            )

        return er_array

    return _erode(imarray, Iterations, Edge_Mode, Constant_Value)


@register_operation(category="morphology", display_name="Closing")
def closing(
    imarray: np.ndarray,
    Iterations: Annotated[int, {"min": 0}] = 1,
    Edge_Mode: Literal[
        "Constant", "Ignore", "Max", "Min", "Mirror", "Nearest", "Reflect", "Wrap"
    ] = "Reflect",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Close an image. Accepts 2D/3D grayscale/RGB images. Dilates then erodes an image.
    Removes small dark spots and connects small bright spots. Convenience wrapper around
    scikit-image's morphology.dilation and morphology.erosion:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion

    Parameters
    ----------
    imarray: np.ndarray
        The image to close.
    Iterations: int
        The number of times to dilate & erode the image. Default is 1.
    Edge_Mode: Literal
        How the algorithm should handle image borders. Default is "Reflect".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Ignore": Pixels outside the border do not influence the result.
        - "Max": Pixels outside the border are set to max dtype value (no influence).
        - "Min": Pixels outside the border are set to min dtype value.
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    cl_array: np.ndarray
        The closed image.
    """

    def _close(imarray, Iterations, Edge_Mode, Constant_Value) -> np.ndarray:

        carray: np.ndarray = np.copy(imarray)
        for _ in range(0, Iterations):
            carray = ski_dilation(carray, mode=Edge_Mode.lower(), cval=Constant_Value)
        for _ in range(0, Iterations):
            carray = ski_erosion(carray, mode=Edge_Mode.lower(), cval=Constant_Value)

        return carray

    if Perform_Slicewise and is3d(imarray):
        cl_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            cl_array[index] = _close(
                imarray[index], Iterations, Edge_Mode, Constant_Value
            )

        return cl_array

    return _close(imarray, Iterations, Edge_Mode, Constant_Value)


@register_operation(category="morphology", display_name="Opening")
def opening(
    imarray: np.ndarray,
    Iterations: Annotated[int, {"min": 0}] = 1,
    Edge_Mode: Literal[
        "Constant", "Ignore", "Max", "Min", "Mirror", "Nearest", "Reflect", "Wrap"
    ] = "Reflect",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Open an image. Accepts 2D/3D grayscale/RGB images. Erodes then dilates an image.
    Removes small bright spots and connects small dark spots. Convenience wrapper around
    scikit-image's morphology.erosion and morphology.dilation:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation

    Parameters
    ----------
    imarray: np.ndarray
        The image to open.
    Iterations: int
        The number of times to erode & dilate the image. Default is 1.
    Edge_Mode: Literal
        How the algorithm should handle image borders. Default is "Reflect".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Ignore": Pixels outside the border do not influence the result.
        - "Max": Pixels outside the border are set to max dtype value (no influence).
        - "Min": Pixels outside the border are set to min dtype value.
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    op_array: np.ndarray
        The opened image.
    """

    def _open(imarray, Iterations, Edge_Mode, Constant_Value) -> np.ndarray:

        oarray: np.ndarray = np.copy(imarray)
        for _ in range(0, Iterations):
            oarray = ski_erosion(oarray, mode=Edge_Mode.lower(), cval=Constant_Value)
        for _ in range(0, Iterations):
            oarray = ski_dilation(oarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        return oarray

    if Perform_Slicewise and is3d(imarray):
        op_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            op_array[index] = _open(
                imarray[index], Iterations, Edge_Mode, Constant_Value
            )

        return op_array

    return _open(imarray, Iterations, Edge_Mode, Constant_Value)


@register_operation(category="morphology", display_name="Black Top Hat")
def black_tophat(
    imarray: np.ndarray,
    Iterations: Annotated[int, {"min": 0}] = 1,
    Edge_Mode: Literal[
        "Constant", "Ignore", "Max", "Min", "Mirror", "Nearest", "Reflect", "Wrap"
    ] = "Reflect",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Perform a black top hat on an image. Accepts 2D/3D grayscale/RGB images. Closes an
    image and then subtracts the original image. Returns small dark areas as bright
    areas. Convenience wrapper around scikit-image's morphology.erosion and
    morphology.dilation:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation

    Parameters
    ----------
    imarray: np.ndarray
        The image to perform the top hat on.
    Iterations: int
        The number of times to dilate & erode the image. Default is 1.
    Edge_Mode: Literal
        How the algorithm should handle image borders. Default is "Reflect".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Ignore": Pixels outside the border do not influence the result.
        - "Max": Pixels outside the border are set to max dtype value (no influence).
        - "Min": Pixels outside the border are set to min dtype value.
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    bth_array: np.ndarray
        The image after the black top hat operation.
    """

    def _bth(imarray, Iterations, Edge_Mode, Constant_Value) -> np.ndarray:

        tharray: np.ndarray = np.copy(imarray)
        for _ in range(0, Iterations):
            tharray = ski_dilation(tharray, mode=Edge_Mode.lower(), cval=Constant_Value)
        for _ in range(0, Iterations):
            tharray = ski_erosion(tharray, mode=Edge_Mode.lower(), cval=Constant_Value)

        return tharray - imarray

    if Perform_Slicewise and is3d(imarray):
        bth_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            bth_array[index] = _bth(
                imarray[index], Iterations, Edge_Mode, Constant_Value
            )

        return bth_array

    return _bth(imarray, Iterations, Edge_Mode, Constant_Value)


@register_operation(category="morphology", display_name="White Top Hat")
def white_tophat(
    imarray: np.ndarray,
    Iterations: Annotated[int, {"min": 0}] = 1,
    Edge_Mode: Literal[
        "Constant", "Ignore", "Max", "Min", "Mirror", "Nearest", "Reflect", "Wrap"
    ] = "Reflect",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Perform a white top hat on an image. Accepts 2D/3D grayscale/RGB images. Opens an
    image and then subtracts it from the original image. Returns small bright areas.
    Convenience wrapper around scikit-image's morphology.dilation and
    morphology.erosion:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion

    Parameters
    ----------
    imarray: np.ndarray
        The image to perform the top hat on.
    Iterations: int
        The number of times to erode & dilate the image. Default is 1.
    Edge_Mode: Literal
        How the algorithm should handle image borders. Default is "Reflect".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Ignore": Pixels outside the border do not influence the result.
        - "Max": Pixels outside the border are set to max dtype value (no influence).
        - "Min": Pixels outside the border are set to min dtype value.
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    wth_array: np.ndarray
        The image after the white top hat operation.
    """

    def _wth(imarray, Iterations, Edge_Mode, Constant_Value) -> np.ndarray:

        tharray: np.ndarray = np.copy(imarray)
        for _ in range(0, Iterations):
            tharray = ski_erosion(tharray, mode=Edge_Mode.lower(), cval=Constant_Value)
        for _ in range(0, Iterations):
            tharray = ski_dilation(tharray, mode=Edge_Mode.lower(), cval=Constant_Value)

        return imarray - tharray

    if Perform_Slicewise and is3d(imarray):
        wth_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            wth_array[index] = _wth(
                imarray[index], Iterations, Edge_Mode, Constant_Value
            )

        return wth_array

    return _wth(imarray, Iterations, Edge_Mode, Constant_Value)


@register_operation(
    category="morphology",
    display_name="Remove Objects",
    layer_type="labels",
    output_kind="labels",
)
def remove_labeled_objects(
    imarray: np.ndarray,
    Method: Literal["Distance", "Size"] = "Size",
    Size: Annotated[int, {"min": 0}] = 10,
    Remove_Above_Size: bool = False,
    Distance: Annotated[float, {"min": 0}] = 10,
) -> np.ndarray:
    """
    Remove objects in a labeled image. Slower than closing/opening/top hat operations
    but more precise. Convenience wrapper around scikit-image's removal functions in the
    morphology package, namely remove_objects_by_distance and remove_small_objects:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_objects_by_distance
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects

    Parameters
    ----------
    imarray: np.ndarray
        The iamge to remove objects from.
    Method: Literal
        The method for removing objects. Default is "Particles".
        - "Distance": Remove from smallest to largest until all are "Distance" apart.
        - "Size": Remove objects in an image smaller than "Size".
    Size: int
        If "Method" is "Size", the minimum pixel count required to keep the object.
    Remove_Above_Size: bool
        If "Method" is "Size", whether to remove objects larger than "Size" (True) or
        smaller than "Size" (False). Default is False.
    Distance: float
        if "Method" is "Distance", the minimum distance between objects required to keep
        both of them.

    Returns
    -------
    rem_array: np.ndarray
        The image after object removal.
    """

    from skimage.morphology import remove_objects_by_distance, remove_small_objects

    if Method == "Distance":
        if Distance == 0:
            return imarray

        return remove_objects_by_distance(imarray, Distance)

    if Size == 0 and not Remove_Above_Size:
        return imarray

    if Remove_Above_Size:
        rem_array: np.ndarray = remove_small_objects(imarray, max_size=Size)
        return imarray - rem_array

    return remove_small_objects(imarray, max_size=Size)


@register_operation(category="morphology", display_name="Remove Holes")
def remove_holes(
    imarray: np.ndarray,
    Size: Annotated[int, {"min": 0}] = 10,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Remove holes in a binary image. Slower than closing/opening/top hat
    operations but more precise. Convenience wrapper around scikit-image's
    morphology.remove_small_holes:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes

    Parameters
    ----------
    imarray: np.ndarray
        The iamge to remove objects from.
    Size: int
        The minimum pixel count required to keep the hole.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    rem_array: np.ndarray
        The image after object removal.
    """

    if Size == 0:
        return imarray

    from skimage.morphology import remove_small_holes

    if Perform_Slicewise and is3d(imarray):
        rem_array: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            rem_array[index] = remove_small_holes(imarray[index], max_size=Size)

        return rem_array

    return remove_small_holes(imarray, max_size=Size)


@register_operation(category="morphology", display_name="Convex Hull")
def convex_hull(
    imarray: np.ndarray,
    Entire_Image: bool = True,
    Connectivity: Literal[1, 2] = 2,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Find the convex hull of a binary image or objects within it. The convex hull is
    the smallest convex polygon that surrounds all white pixels in the image/object.
    Convenience wrapper around scikit-image's morphology.convex_hull_image and
    morphology.convex_hull_object:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.convex_hull_image
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.convex_hull_object

    Parameters
    ----------
    imarray: np.ndarray
        The image to find the convex hull for.
    Entire_Image: bool
        Whether to find the convex hull of the entire image (True) or individual objects
        (False). Default is True. If False, operation is performed slicewise for 3D.
    Connectivity: bool
        If "Entire_Image" is False, the connecitivity to use for labeling different
        objects.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    convex_array: np.ndarray
        The convex hull image.
    """

    if isrgb(imarray):
        print("Image is RGB, convex hull not found!")
        return imarray

    from skimage.morphology import convex_hull_image, convex_hull_object

    def _convex_hull(imarray, Entire_Image, Connectivity):
        if Entire_Image:
            return convex_hull_image(imarray)
        return convex_hull_object(imarray, connectivity=Connectivity)

    if (Perform_Slicewise or not Entire_Image) and is3d(imarray):
        convex_array: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            convex_array[index] = _convex_hull(
                imarray[index], Entire_Image, Connectivity
            )

        return convex_array

    return _convex_hull(imarray, Entire_Image, Connectivity)


@register_operation(category="morphology", display_name="Thin")
def thin(
    imarray: np.ndarray,
    Fully_Skeletonize: bool = True,
    Skeletonize_Method: Literal["Lee", "Zhang"] = "Lee",
    Thinning_Iterations: Annotated[int, {"min": 0}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Thin a binary image without affecting its connectivity. Convenience wrapper around
    scikit-image's morphology.thin and morphology.skeletonize:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.thin
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize

    Parameters
    ----------
    imarray: np.ndarray
        The image array to thin.
    Fully_Skeletonize: bool
        Whether to fully skeletonize the image (True) or just thin it (False). If False,
        performed slicewise for 3D. Default is True.
    Skeletonize_Method: Literal
        If "Fully Skeletonize" is True, the skeletonization method to use.
        Default is "Lee".
        - "Lee": Can handle 2D/3D.
        - "Zhang": 2D only, performed slicewise for 3D.
    Thinning_Iterations: int
        If "Fully Skeletonize" is False, the max number of iterations that can be used
        to thin the image. if 0, no restriction is put on iterations.
    Perform_Slicewise: bool
        Whether to performt the operation slicewise (True) or in bulk (False). Only
        applies if "Skeletonize Method" is "Lee" and "Fully Skeletonize" is True.
        Default is False.

    Returns
    -------
    thin_array: np.ndarray
        The thinned image.
    """

    if isrgb(imarray):
        print("Image is RGB, thinning not performed!")
        return imarray

    from skimage.morphology import skeletonize
    from skimage.morphology import thin as ski_thin

    def _thin(
        imarray, Fully_Skeletonize, Skeletonize_Method, Thinning_Iterations
    ) -> np.ndarray:

        if Fully_Skeletonize:
            return skeletonize(imarray, method=Skeletonize_Method.lower())

        return ski_thin(imarray, Thinning_Iterations)

    if is3d(imarray) and (
        Skeletonize_Method == "Zhang" or not Fully_Skeletonize or Perform_Slicewise
    ):
        thin_array: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            thin_array[index] = _thin(
                imarray[index],
                Fully_Skeletonize,
                Skeletonize_Method,
                Thinning_Iterations,
            )

        return thin_array

    return _thin(imarray, Fully_Skeletonize, Skeletonize_Method, Thinning_Iterations)


@register_operation(category="morphology", display_name="Extrema")
def extrema(
    imarray: np.ndarray,
    Maxima: bool = True,
    Minimum_Difference: Annotated[int, {"min": 0}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Find the local extrema in an image based on morphological connectivity. Accepts
    grayscale 2D/3D images. Convenience wrapper around scikit-image's morphology.extema
    functions, namely h_maxima, h_minima, local_maxima, and local_minima:
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.h_maxima
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.h_minima
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.local_maxima
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.local_minima

    Parameters
    ----------
    imarray: np.ndarray
        The image to find local extrema in.
    Maxima: bool
        Whether to find local maxima (True) or minima (False). Default is True (maxima).
    Minimum_Difference: int
        The minimum difference between the local maxima and its surroundings. If 0, this
        value is ignored.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    exarray: np.ndarray
        Boolean array with local extrema indicated as True.
    """

    if isrgb(imarray):
        print("Image is RGB, local extrema not found!")
        return imarray

    from skimage.morphology import h_maxima, h_minima, local_maxima, local_minima

    def _extrema(imarray, Maxima, Minimum_Difference) -> np.ndarray:

        if Minimum_Difference:
            if Maxima:
                return h_maxima(imarray, Minimum_Difference).astype(np.bool)
            return h_minima(imarray, Minimum_Difference).astype(np.bool)
        if Maxima:
            return local_maxima(imarray)  # type: ignore
        return local_minima(imarray)  # type: ignore

    if Perform_Slicewise and is3d(imarray):
        exarray: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            exarray[index] = _extrema(imarray[index], Maxima, Minimum_Difference)

        return exarray

    return _extrema(imarray, Maxima, Minimum_Difference)
