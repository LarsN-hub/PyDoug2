"""
Domain transformations of images.
"""

import numpy as np
from skimage import transform

from pydoug_core import util


def radon(imarray: np.ndarray) -> np.ndarray:
    """
    Transform an image into the projection domain.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.

    Returns
    -------
    radarray: np.ndarray
        The image array transformed into the projection domain.
    """

    imarray = util.rgb2gray(imarray)
    if util.is3d(imarray):
        firstrad: np.ndarray = transform.radon(imarray[0])
        radarray: np.ndarray = np.empty(
            (imarray.shape[0], firstrad.shape[0], firstrad.shape[1])
        )
        for index in range(0, imarray.shape[0]):
            radarray[index] = transform.radon(imarray[index])
        return radarray
    else:
        return transform.radon(imarray)
