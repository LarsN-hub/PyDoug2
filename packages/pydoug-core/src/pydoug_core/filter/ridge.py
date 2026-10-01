"""
Filters that detect ridges.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.filters import frangi, hessian, meijering, sato

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["ridge"]


@register_operation(category="filters", display_name="Ridges")
def ridge(
    imarray: np.ndarray,
    Method: Literal["Frangi", "Hessian", "Meijering", "Sato"] = "Sato",
    Sigmas: tuple[
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 1}],
    ] = (1, 10, 2),
    Alpha: Annotated[float, {"min": 0}] = 0.5,
    Beta: Annotated[float, {"min": 0}] = 0.5,
    Gamma: Annotated[float, {"min": 0}] = 0,
    Black_Ridges: bool = True,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Filter an image with a ridge response filter. Ridges are similar to edges, but have
    another edge nearby (e.g., a vein, tubule, or crack). Convenience wrapper around the
    ridge filters in scikit-image's filter package, including frangi, hessian,
    meijering, and sato:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.hessian
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.meijering
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato

    Parameters
    ----------
    imarray: np.ndarray
        The image to filter.
    Method: Literal["Frangi", "Hessian", "Meijering", "Sato"]
        The ridge filter to use. All filters work in 2D/3D but are applied channel-wise
        if the image is RGB. Default is Sato.
        - "Frangi": Filter using the Frangi vesselness filter.
        - "Hessian": Filter using the hybrid Hessian filter.
        - "Meijering": Filter with the Meijering neuriteness filter.
        - "Sato": Filter with the Sato tubeness filter.
    Sigmas: tuple[int, int, int]
        The range of standard deviations to optimize the filter. Provided as:
        - Left-Most: Starting sigma.
        - Middle: Ending sigma (must be >= starting sigma).
        - Right-Most: Step interval between starting and ending sigma (must be +).
    Alpha: float
        Frangi correction factor for plate structures if "Method" is "Frangi" or
        "Hessian", or the shaping filter constant if "Method" is "Meijering".
        Ignored for "Sato".
    Beta: float
        Frangi correction factor for blob structures if "Method" is "Frangi" or
        "Hessian". Ignored for other methods.
    Gamma: float
        Frangi correction factor for high variance areas if "Method" is "Frangi" or
        "Hessian". Ignored for other methods. Automatically calculated if 0.
    Black_Ridges: bool
        Whether the ridges are darker than the surroundings or not. Default is True.
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        How the algorithm should handle image borders. Ignored if "Method" is "Laplace"
        or "Roberts". Ignored if "Horizontal" or "Vertical" are True.
        Default is "Nearest".
        - "Constant": The border is extended with a value defined in "Constant Value".
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
    ridge_array: np.ndarray
        The image after ridge filtering.
    """

    def _ridge(
        imarray,
        Method,
        Sigmas,
        Alpha,
        Beta,
        Gamma,
        Black_Ridges,
        Edge_Mode,
        Constant_Value,
    ) -> np.ndarray:

        if Method == "Frangi":
            if Gamma == 0:
                gamma = None
            else:
                gamma = Gamma
            if util.isrgb(imarray):
                ridge_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    ridge_array[..., cindex] = frangi(
                        imarray[..., cindex],
                        range(Sigmas[0], Sigmas[1], Sigmas[2]),
                        alpha=Alpha,
                        beta=Beta,
                        gamma=gamma,
                        black_ridges=Black_Ridges,
                        mode=Edge_Mode.lower(),
                        cval=Constant_Value,
                    )

                return ridge_array

            return frangi(
                imarray,
                range(Sigmas[0], Sigmas[1], Sigmas[2]),
                alpha=Alpha,
                beta=Beta,
                gamma=gamma,
                black_ridges=Black_Ridges,
                mode=Edge_Mode.lower(),
                cval=Constant_Value,
            )

        elif Method == "Hessian":
            if Gamma == 0:
                gamma = 15
            else:
                gamma = int(round(Gamma))
            if util.isrgb(imarray):
                ridge_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    ridge_array[..., cindex] = hessian(
                        imarray[..., cindex],
                        range(Sigmas[0], Sigmas[1], Sigmas[2]),
                        alpha=Alpha,
                        beta=Beta,
                        gamma=gamma,
                        black_ridges=Black_Ridges,
                        mode=Edge_Mode.lower(),
                        cval=Constant_Value,
                    )

                return ridge_array

            return hessian(
                imarray,
                range(Sigmas[0], Sigmas[1], Sigmas[2]),
                alpha=Alpha,
                beta=Beta,
                gamma=gamma,
                black_ridges=Black_Ridges,
                mode=Edge_Mode.lower(),
                cval=Constant_Value,
            )

        elif Method == "Meijering":
            if util.isrgb(imarray):
                ridge_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    ridge_array[..., cindex] = meijering(
                        imarray[..., cindex],
                        range(Sigmas[0], Sigmas[1], Sigmas[2]),
                        Alpha,
                        Black_Ridges,
                        Edge_Mode.lower(),
                        Constant_Value,
                    )

                return ridge_array

            return meijering(
                imarray,
                range(Sigmas[0], Sigmas[1], Sigmas[2]),
                Alpha,
                Black_Ridges,
                Edge_Mode.lower(),
                Constant_Value,
            )

        else:
            if util.isrgb(imarray):
                ridge_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    ridge_array[..., cindex] = sato(
                        imarray[..., cindex],
                        range(Sigmas[0], Sigmas[1], Sigmas[2]),
                        Black_Ridges,
                        Edge_Mode.lower(),
                        Constant_Value,
                    )

                return ridge_array

            return sato(
                imarray,
                range(Sigmas[0], Sigmas[1], Sigmas[2]),
                Black_Ridges,
                Edge_Mode.lower(),
                Constant_Value,
            )

    if Perform_Slicewise and util.is3d(imarray):
        ridge_array: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            ridge_array[index] = _ridge(
                imarray[index],
                Method,
                Sigmas,
                Alpha,
                Beta,
                Gamma,
                Black_Ridges,
                Edge_Mode,
                Constant_Value,
            )

        return ridge_array

    return _ridge(
        imarray,
        Method,
        Sigmas,
        Alpha,
        Beta,
        Gamma,
        Black_Ridges,
        Edge_Mode,
        Constant_Value,
    )
