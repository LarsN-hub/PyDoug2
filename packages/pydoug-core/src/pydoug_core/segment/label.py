"""
Labeling algorithms for segmented images.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.measure import label as conn_label

from pydoug_core.registry import register_operation

__all__ = ["connectivity"]


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
