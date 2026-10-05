"""
Labeling algorithms for segmented images.
"""

from typing import Annotated, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label as conn_label
from skimage.morphology import ball, disk
from skimage.segmentation import watershed as ws

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["connectivity", "watershed"]


@register_operation(
    category="segmentation",
    display_name="Connectivity Label",
    output_kind="labels",
)
def connectivity(
    imarray: np.ndarray,
    Connectivity: Literal[1, 2, 3] = 2,
    Background: Annotated[int, {"min": 0}] = 0,
) -> np.ndarray:
    """
    Label an image using pixel connectivity. Convenience wrapper around scikit-image's
    measure.label:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

    Parameters
    ----------
    imarray: np.ndarray
        The segmented image to be labeled.
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
    Background: int
        The background value that should not be included in the labeling process.
        Default is 0.

    Returns
    -------
    lab_array: np.ndarray
        The labeled segmented image.
    """

    return conn_label(imarray, Background, connectivity=Connectivity)  # type: ignore


@register_operation(
    category="segmentation",
    display_name="Watershed Label",
    array_params=("Markers",),
    array_param_types={"Markers": "labels"},
    array_param_flags={
        "Markers": "User_Defined_Markers",
    },
    output_kind="labels",
)
def watershed(
    imarray: np.ndarray,
    Markers: np.ndarray | None = None,
    User_Defined_Markers: bool = False,
    Use_Negative_Image: bool = True,
    Automatic_Radius: Annotated[int, {"min": 1}] = 3,
    Connectivity: Literal[1, 2, 3] = 1,
    Compactness: Annotated[float, {"min": 0}] = 0,
    Watershed_Line: bool = False,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Label an image with a watershed algorithm. Markers can be provided or automatically
    generated. Can be performed on 2D/3D grayscale images only.

    Parameters
    ----------
    imarray: np.ndarray
        The image to be labeled.
    Markers: np.ndarray
        The user-defined markers to mark the watershed basins. Ignored if
        "User_Defined_Markers" is False.
    User_Defined_Markers: bool
        Whether or not to use "Markers" (True) or attempt to automatically generate
        markers (False). Default is False (requires "Markers").
        If False, the image is assumed to be binary and markers are automatically
        generated using scipy's ndimage.distance_transform_edt and scikit-image's
        feature.peak_local_max as in this method:
        https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#
    Use_Negative_Image: bool
        Whether or not to use the negative of the input image. Default is True.
        Ignored if "User_Defined_Markers" is False.
    Automatic_Radius: bool
        If "User_Defined_Markers" is False, the radius to use for the peak_local_max
        markers generation.
    Connectivity: Literal[1, 2, 3]
        The method to use for determing label connectivity. Default is 1.
        - 1: Only pixels with faces in contact are considered connected.
            - In 2D, this means only four most adjacent pixels.
            - In 3D, this means only the 6 most adjacent voxels.
        - 2: Includes 1, and adds pixels with edges in contact.
            - In 2D, this means the surrounding 8 pixels.
            - In 3D, this means only the 18 most adjacent voxels.
        - 3: (3D only) Includes 1-2, and adds pixels with corners in contact.
            - In 3D, this means the surrounding 26 voxels.
    Compactness: float
        The compactness of the watershed labeling. If > 0, compact watershed labeling is
        performed.
    Watershed_Line: bool
        Whether or not to separate the watershed domains with a background line.
        Default is False.
    Perform_Slicewise: bool
        Whether to perform the operation slicewise (True) or in bulk (False).
        Default is False.

    Returns
    -------
    wsarray: np.ndarray
        The image after watershed labeling.
    """

    if util.isrgb(imarray):
        print("Image is RGB, watershed not performed!")
        return imarray

    if Connectivity > 2 and Perform_Slicewise or not util.is3d(imarray):
        Connectivity = 2

    def _watershed(
        imarray,
        Markers,
        User_Defined_Markers,
        Use_Negative_Image,
        Automatic_Radius,
        Connectivity,
        Compactness,
        Watershed_Line,
    ) -> np.ndarray:

        if User_Defined_Markers:
            if Use_Negative_Image:
                return ws(
                    -imarray,
                    Markers,
                    Connectivity,
                    mask=imarray.astype(np.bool),
                    compactness=Compactness,
                    watershed_line=Watershed_Line,
                )
            else:
                return ws(
                    imarray,
                    Markers,
                    Connectivity,
                    mask=imarray.astype(np.bool),
                    compactness=Compactness,
                    watershed_line=Watershed_Line,
                )

        if util.is3d(imarray):
            footprint = ball(Automatic_Radius)
        else:
            footprint = disk(Automatic_Radius)
        distance: np.ndarray = distance_transform_edt(imarray)  # type: ignore
        peak_coords: np.ndarray = peak_local_max(distance, footprint=footprint)
        water_mask_array: np.ndarray = np.zeros(distance.shape, dtype=np.bool)
        water_mask_array[tuple(peak_coords.T)] = True
        markers: np.ndarray = conn_label(water_mask_array)  # type: ignore

        return ws(
            -distance,
            markers,
            Connectivity,
            mask=imarray,
            compactness=Compactness,
            watershed_line=Watershed_Line,
        )

    if Perform_Slicewise and util.is3d(imarray):
        wsarray: np.ndarray = np.empty(imarray.shape, np.int32)
        for index in range(0, imarray.shape[0]):
            if User_Defined_Markers and Markers is not None:
                markers: np.ndarray | None = Markers[index]
            else:
                markers = None
            wsarray[index] = _watershed(
                imarray[index],
                markers,
                User_Defined_Markers,
                Use_Negative_Image,
                Automatic_Radius,
                Connectivity,
                Compactness,
                Watershed_Line,
            )

        return wsarray

    return _watershed(
        imarray,
        Markers,
        User_Defined_Markers,
        Use_Negative_Image,
        Automatic_Radius,
        Connectivity,
        Compactness,
        Watershed_Line,
    )
