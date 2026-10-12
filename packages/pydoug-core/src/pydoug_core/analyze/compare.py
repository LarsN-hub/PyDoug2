"""
Compare two image datasets.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["mse", "ssi", "psnr", "are", "voi", "nmi", "intersection", "hausdorff_dist"]


@register_operation(
    category="compare",
    display_name="Mean Squared Error",
    array_params=("Layer2",),
    output_kind="value",
)
def mse(
    imarray: np.ndarray,
    Layer2: np.ndarray,
    NRMSE: bool = False,
    Normalization_Method: Literal["Euclidean", "Min-Max", "Mean"] = "Euclidean",
) -> float:
    """
    Calculate the mean squared error (MSE) between two images. Images must have the same
    dimensions but can be 2D/3D grayscale/RGB. Useful for judging intensity differences
    between images. Ranges from 0 (perfectly identical images) upwards (higher = less
    similar). Convenience wrapper around scikit-image's metrics.mean_squared_error and
    metrics.normalized_root_mse:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.mean_squared_error
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.normalized_root_mse

    Parameters
    ----------
    imarray: np.ndarray
        The ground-truth image.
    Layer2: np.ndarray
        The image to test against the ground-truth image.
    NRMSE: bool
        Whether to compute the normalized root MSE (True) or standard MSE (False).
        Default is False.
    Normalization_Method: Literal
        The normalization method to use if "NRMSE" is "True". Ignored otherwise. Default
        is "Euclidean".
        - "Euclidean": Normalize by averaged Euclidean norm of ground-truth image.
        - "Min-Max": Normalize by intensity range of ground-truth image.
        - "Mean": Normalize by mean of ground-truth image.

    Returns
    -------
    mse_val: float
        The MSE between the two images.
    """

    if NRMSE:
        from skimage.metrics import normalized_root_mse

        return normalized_root_mse(
            imarray, Layer2, normalization=Normalization_Method.lower()
        )

    from skimage.metrics import mean_squared_error

    return mean_squared_error(imarray, Layer2)


@register_operation(
    category="compare",
    display_name="Structural Similarity",
    array_params=("Layer2",),
    output_kind="value",
)
def ssi(
    imarray: np.ndarray,
    Layer2: np.ndarray,
) -> float:
    """
    Calculate the structural similarity index (SSI) between two images. Images must have
    the same dimensions but can be 2D/3D grayscale/RGB. Useful for judging intensity and
    texture differences between images. Ranges from 1 (identical images) to 0 (perfectly
    non-similar images). Convenience wrapper around scikit-image's
    metrics.structural_similarity:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

    Parameters
    ----------
    imarray: np.ndarray
        The first image.
    Layer2: np.ndarray
        The second image.

    Returns
    -------
    ssi_val: float
        The SSI between the two images.
    """

    from skimage.metrics import structural_similarity

    from pydoug_core.util import chan_axis

    if imarray.dtype == np.bool:
        imrange: float | int = 1
    else:
        imrange = np.max(imarray) - np.min(imarray)

    return structural_similarity(
        imarray,
        Layer2,
        data_range=imrange,
        channel_axis=chan_axis(imarray),
    )  # type: ignore


@register_operation(
    category="compare",
    display_name="Peak Signal to Noise Ratio",
    array_params=("Layer2",),
    output_kind="value",
)
def psnr(
    imarray: np.ndarray,
    Layer2: np.ndarray,
) -> float:
    """
    Calculate the peak signal to noise ratio (PSNR) between two images. Images must be
    the same dimension but can be 2D/3D grayscale/RGB. Ratio of the peak signal of an
    image to the corrupting noise present (higher = better, PSNR -> infinity if both
    images are identical). Convenience wrapper around scikit-image's
    metrics.peak_signal_noise_ratio:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio

    Parameters
    ----------
    imarray: np.ndarray
        The ground-truth image.
    Layer2: np.ndarray
        The image to test against the ground-truth image.

    Returns
    -------
    psnr_val: float
        The PSNR between the two images.
    """

    from skimage.metrics import peak_signal_noise_ratio

    if imarray.dtype == np.bool:
        imrange: float | int = 1
    else:
        imrange = np.max(imarray) - np.min(imarray)

    return peak_signal_noise_ratio(imarray, Layer2, data_range=imrange)


@register_operation(
    category="compare",
    display_name="Adapted Rand Error",
    array_params=("Layer2",),
    output_kind="value",
)
def are(
    imarray: np.ndarray,
    Layer2: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the adapted Rand error between an image and a ground-truth image. Images
    must have the same dimensions but can be 2D/3D grayscale/RGB. Useful for evaluating
    segmentation similarity. Returns "Error", "Precision", and "Recall", where "Error"
    indicates overall similarity (0 = perfectly similar). "Precision" is the number of
    correctly segmented pixels divided by the number of segmented pixels in the test
    image and "Recall" is the number of correctly segmented pixels divided by the number
    of segmented pixels in the true image. If "Precision" is close to 1 but "Recall" is
    lower, the image is undersegmented. If "Recall" is close to 1 but "Precision" is
    lower, the image is oversegmented. Both are 1 for perfect segmentation. Non-integer
    images are converted before calculation. Convenience wrapper around scikit-image's
    metrics.adapted_rand_error:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.adapted_rand_error

    Parameters
    ----------
    imarray: np.ndarray
        The ground truth image.
    Layer2: np.ndarray
        The image to test against the ground truth image.

    Returns
    -------
    are_dict:
        The adapted Rand error, precision, and recall between the two images.
    """

    from skimage.metrics import adapted_rand_error

    are, prec, rec = adapted_rand_error(
        np.round(imarray).astype(np.int32), np.round(Layer2).astype(np.int32)
    )

    return {"Error": are, "Precision": prec, "Recall": rec}


@register_operation(
    category="compare",
    display_name="Variation of Information",
    array_params=("Layer2",),
    output_kind="value",
)
def voi(
    imarray: np.ndarray,
    Layer2: np.ndarray,
    Rounding_Decimals: Annotated[int, {"min": 0}] = 5,
) -> dict[str, float]:
    """
    Calculate the variation of information (VOI) between two images. Images must have
    the same dimensions but can be 2D/3D grayscale/RGB. Useful for assessing over/under
    segmentation with a ground-truth image. Returns "Splits" and "Merges" where "Splits"
    indicates undersegmentation while "Merges" indicates oversegmentation. 0 = perfect
    similarity while higher means less similar. Non-integer images are converted before
    calculation. Convenience wrapper around scikit-image's
    metrics.variation_of_information:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.variation_of_information

    Parameters
    ----------
    imarray: np.ndarray
        The ground-truth image.
    Layer2: np.ndarray
        The image to compare to the ground-truth image.
    Rounding_Decimals: int
        The number of decimal points to round to (returns 0 if truly perfect instead of
        just a very small number). Default is 5, increase for more decimals points.

    Returns
    -------
    voi_vals: dict[str, float]
        A dictionary containing the splits and merges from the VOI calculation.
    """

    from skimage.metrics import variation_of_information

    splits, merges = variation_of_information(
        np.round(imarray).astype(np.int32),
        np.round(Layer2).astype(np.int32),
    )

    return {
        "Splits": round(splits, Rounding_Decimals),
        "Merges": round(merges, Rounding_Decimals),
    }


@register_operation(
    category="compare",
    display_name="Normalized Mutual Information",
    array_params=("Layer2",),
    output_kind="value",
)
def nmi(
    imarray: np.ndarray,
    Layer2: np.ndarray,
    Bin_Count: Annotated[int, {"min": 1}] = 100,
    Rounding_Decimals: Annotated[int, {"min": 0}] = 5,
) -> float:
    """
    Calculate the normalized mutual information (NMI) between two images. Images must
    have the same dimensions but can be 2D/3D grayscale/RGB. Useful for assessing image
    registration quality (higher = better). Measured from 1 to 2 where 2 = perfect
    correlation and 1 = perfect non-correlation. Convenience wrapper around
    scikit-image's metrics.normalized_mutual_information:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.normalized_mutual_information

    Parameters
    ----------
    imarray: np.ndarray
        The first image.
    Layer2: np.ndarray
        The second image.
    Bin_Count: int
        The number of bins to use to calculate the joint histogram.
    Rounding_Decimals: int
        The number of decimal points to round to (returns 0 if truly perfect instead of
        just a very small number). Default is 5, increase for more decimals points.

    Returns
    -------
    nmi_val: float
        The NMI between the two images.
    """

    from skimage.metrics import normalized_mutual_information

    return round(
        normalized_mutual_information(imarray, Layer2, bins=Bin_Count),  # type: ignore
        Rounding_Decimals,
    )


@register_operation(
    category="compare",
    display_name="Intersection Coefficient",
    array_params=("Layer2",),
    output_kind="value",
)
def intersection(
    imarray: np.ndarray,
    Layer2: np.ndarray,
) -> float:
    """
    Determine the fraction of the segmented pixels in the first image that lie within
    the segmented pixels of the second image. Accepts 2D/3D grayscale/RGB images but is
    only meaningful for binary segmented images. All nonzero pixels in non-binary images
    are treated as segmented pixels. Convenience wrapper around numpy's boolean indexing
    and count_nonzero:
    https://numpy.org/doc/1.21/reference/arrays.indexing.html#boolean-array-indexing
    https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html

    Parameters
    ----------
    imarray: np.ndarray
        The first image.
    Layer2: np.ndarray
        The second image.

    Returns
    -------
    inter_coeff: float
        The intersection coefficient between the two images.
    """

    from pydoug_core.util import convert_dtype, rgb2gray

    one_total: np.intp = np.count_nonzero(convert_dtype(rgb2gray(imarray), "Bool"))
    one_count: np.intp = np.count_nonzero(
        convert_dtype(rgb2gray(imarray), "Bool")[
            convert_dtype(rgb2gray(Layer2), "Bool")
        ]
    )

    return one_count / one_total


@register_operation(
    category="compare",
    display_name="Hausdorff Distance",
    array_params=("Layer2",),
    output_kind="value",
)
def hausdorff_dist(
    imarray: np.ndarray,
    Layer2: np.ndarray,
) -> float:
    """
    Calculate the Hausdorff distance between two images (the maximum distance between
    a nonzero pixel in the first image and the nearest nonzero pixel in the second
    image). Images must have the same dimensions but can be 2D/3D grayscale/RGB.
    Convenience wrapper around scikit-image's metrics.hausdorff_distance:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.hausdorff_distance

    Parameters
    ----------
    imarray: np.ndarray
        The first image.
    Layer2: np.ndarray
        The second image.

    Returns
    -------
    hd_val: float
        The Hausdorff distance between the two images.
    """

    from skimage.metrics import hausdorff_distance

    return hausdorff_distance(imarray, Layer2)  # type: ignore
