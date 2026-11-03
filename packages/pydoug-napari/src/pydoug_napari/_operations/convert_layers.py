"""
Convert layer types as able.
"""

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["lab2im", "im2lab"]


@register_operation(
    category="visualize",
    display_name="Labels to Image",
    layer_type="labels",
)
def lab2im(labarray: np.ndarray) -> np.ndarray:
    """
    Convert a labels layer to an image layer.

    Parameters
    ----------
    labarray: np.ndarray
        The labeled image.

    Returns
    -------
    imarray: np.ndarray
        The image array.
    """

    return labarray


@register_operation(
    category="visualize",
    display_name="Image to Labels",
    output_kind="labels",
)
def im2lab(imarray: np.ndarray) -> np.ndarray:
    """
    Convert an image layer to a labels layer. Input image must have an integer dtype.
    If image is RGB it will first be converted to grayscale.

    Parameters
    ----------
    imarray: np.ndarray
        The image array.

    Returns
    -------
    labarray: np.ndarray
        The labeled image.
    """

    if np.issubdtype(imarray.dtype, np.floating):
        raise ValueError("Image does not have an integer dtype!")

    from pydoug_core.util import isrgb

    if isrgb(imarray):
        from pydoug_core.util import gray2rgb

        return gray2rgb(imarray)

    return imarray
