"""
Filters that detect gradients and edges.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["gradient", "ridge"]


@register_operation(category="filter", display_name="Gradient")
def gradient(
    imarray: np.ndarray,
    Method: Literal[
        "Farid", "Laplace", "Local", "Prewitt", "Roberts", "Scharr", "Sobel"
    ] = "Sobel",
    K_Size: Annotated[int, {"min": 1}] = 3,
    Radius: Annotated[int, {"min": 1}] = 3,
    Horizontal: bool = False,
    Vertical: bool = False,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Filter an image with a gradient response filter. Convenience wrapper around the
    gradient filters in scikit-image's filters package, including farid, laplace,
    prewitt, roberts, scharr, and sobel, as well as the (local) gradient function in
    scikit-image's filters.rank package:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_v
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.laplace
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_v
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts_neg_diag
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts_pos_diag
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_v
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_v
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.gradient

    Parameters
    ----------
    imarray: np.ndarray
        The image to be filtered.
    Method: Literal[
        "Farid", "Laplace", "Local", "Prewitt", "Roberts", "Scharr", "Sobel"
    ]
        The method to use to calculate the gradient response. "Roberts" will
        automatically be performed slicewise if the input image is 3D. "Local" is the
        only one that requiers "Radius".
        Default is "Sobel".
    K_Size: int
        The k size if "Method" is "Laplace". Ignored for all other methods.
    Radius: int
        The size of the local radius if "Method" is "Local". Ignored for all others.
    Horizontal: bool
        Whether or not to apply the filter horizontally only. "Vertical" must be False
        or this will be ignored. Forces slicewise performance on 3D images and channel-
        wise performance on RGB images. Performs the negative diagonal if "Method" is
        "Roberts".
    Vertical: bool
        Whether or not to apply the filter vertically only. "Horizontal" must be False
        or this will be ignored. Forces slicewise performance on 3D images and channel-
        wise performance on RGB images. Performs the positive diagonal if "Method" is
        "Roberts".
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
    grad_array: np.ndarray
        The gradient filtered image.
    """

    from skimage.filters import (
        farid,
        farid_h,
        farid_v,
        laplace,
        prewitt,
        prewitt_h,
        prewitt_v,
        roberts,
        roberts_neg_diag,
        roberts_pos_diag,
        scharr,
        scharr_h,
        scharr_v,
        sobel,
        sobel_h,
        sobel_v,
    )
    from skimage.filters.rank import gradient as rank_grad
    from skimage.morphology import ball, disk

    from pydoug_core.util import is3d, isrgb

    def _grad(
        imarray, Method, K_Size, Radius, Horizontal, Vertical, Edge_Mode, Constant_Value
    ) -> np.ndarray:

        if Method == "Farid":
            if Horizontal and not Vertical:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = farid_h(imarray[..., cindex])
                    return grad_array
                return farid_h(imarray)
            elif Vertical and not Horizontal:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = farid_v(imarray[..., cindex])
                    return grad_array
                return farid_v(imarray)
            return farid(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        elif Method == "Laplace":
            return laplace(imarray, K_Size)

        elif Method == "Local":
            if is3d(imarray):
                fp = ball(Radius)
            else:
                fp = disk(Radius)
            if isrgb(imarray):
                grad_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    grad_array[..., cindex] = rank_grad(imarray[..., cindex], fp)
                return grad_array
            return rank_grad(imarray, fp)

        elif Method == "Prewitt":
            if Horizontal and not Vertical:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = prewitt_h(imarray[..., cindex])
                    return grad_array
                return prewitt_h(imarray)
            elif Vertical and not Horizontal:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = prewitt_v(imarray[..., cindex])
                    return grad_array
                return prewitt_v(imarray)
            return prewitt(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        elif Method == "Roberts":
            if Horizontal and not Vertical:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = roberts_neg_diag(imarray[..., cindex])
                    return grad_array
                return roberts_neg_diag(imarray)
            elif Vertical and not Horizontal:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = roberts_pos_diag(imarray[..., cindex])
                    return grad_array
                return roberts_pos_diag(imarray)
            if isrgb(imarray):
                grad_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    grad_array[..., cindex] = roberts(imarray[..., cindex])
                return grad_array
            return roberts(imarray)

        elif Method == "Scharr":
            if Horizontal and not Vertical:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = scharr_h(imarray[..., cindex])
                    return grad_array
                return scharr_h(imarray)
            elif Vertical and not Horizontal:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = scharr_h(imarray[..., cindex])
                    return grad_array
                return scharr_v(imarray)
            return scharr(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        else:
            if Horizontal and not Vertical:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = sobel_h(imarray[..., cindex])
                    return grad_array
                return sobel_h(imarray)
            elif Vertical and not Horizontal:
                if isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[..., cindex] = sobel_v(imarray[..., cindex])
                    return grad_array
                return sobel_v(imarray)
            return sobel(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

    if Method == "Roberts":
        Perform_Slicewise = True
    if Horizontal and not Vertical:
        Perform_Slicewise = True
    if Vertical and not Horizontal:
        Perform_Slicewise = True

    if Perform_Slicewise and is3d(imarray):
        grad_array: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            grad_array[index] = _grad(
                imarray[index],
                Method,
                K_Size,
                Radius,
                Horizontal,
                Vertical,
                Edge_Mode,
                Constant_Value,
            )
        return grad_array
    else:
        return _grad(
            imarray,
            Method,
            K_Size,
            Radius,
            Horizontal,
            Vertical,
            Edge_Mode,
            Constant_Value,
        )


@register_operation(category="filter", display_name="Ridges")
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
        How the algorithm should handle image borders. Default is "Nearest".
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

    from skimage.filters import frangi, hessian, meijering, sato

    from pydoug_core.util import is3d, isrgb

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
            if isrgb(imarray):
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
            if isrgb(imarray):
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
            if isrgb(imarray):
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
            if isrgb(imarray):
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

    if Perform_Slicewise and is3d(imarray):
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
