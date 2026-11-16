"""
Image registration.
"""

from typing import Annotated, Literal

import numpy as np
from SimpleITK import Transform as sitkTransform

from pydoug_core import util
from pydoug_core.registry import register_operation

_KINDS = ("Translation", "Rigid", "Similarity", "Affine")
_INTERPOLATORS = {
    "Linear": "sitkLinear",
    "Nearest": "sitkNearestNeighbor",
    "BSpline": "sitkBSpline",
}


def _make_transform(kind: str, dim: int) -> sitkTransform:
    """
    Return the SimpleITK transform needed for the input kind and dimensionality.

    Parameters
    ----------
    kind: str
        The kind of transform being performed.
    dim: int
        The dimensions of the image array.

    Returns
    -------
    sitk_trans: Transform
        The SimpleITK transform.
    """

    import SimpleITK as sitk

    if kind == "Translation":
        return sitk.TranslationTransform(dim)
    if kind == "Rigid":
        return sitk.Euler2DTransform() if dim == 2 else sitk.Euler3DTransform()
    if kind == "Similarity":
        return (
            sitk.Similarity2DTransform() if dim == 2 else sitk.Similarity3DTransform()
        )
    return sitk.AffineTransform(dim)


def _pack(kind: str, transform: sitkTransform) -> np.ndarray:
    """
    Return a numpy array with the kind index, dimension, center size, center, and
    parameters of the transform.

    Parameters
    ----------
    kind: str
        The kind of transform being performed.
    transform: Transform
        The SimpleITK transform.

    Returns
    -------
    pack_array: np.ndarray
        The numpy array.
    """

    fixed: list = list(transform.GetFixedParameters())
    parameters: list = list(transform.GetParameters())
    return np.array(
        [
            [
                _KINDS.index(kind),
                transform.GetDimension(),
                len(fixed),
                *fixed,
                *parameters,
            ]
        ],
        dtype=float,
    )


def _unpack(vector: np.ndarray) -> sitkTransform:
    """
    Unpack a packed numpy array created with _pack.

    Parameters
    ----------
    vector: np.ndarray
        The packed numpy array.

    Returns
    -------
    trans: Transform
        The SimpleITK transform.
    """

    values: np.ndarray = np.asarray(vector, dtype=float).ravel()
    message: str = "The Transform layer was not made by Compute Registration."
    if values.size < 3 or not 0 <= values[0] < len(_KINDS):
        raise ValueError(message)
    try:
        n_fixed: int = int(values[2])
        transform: sitkTransform = _make_transform(
            _KINDS[int(values[0])], int(values[1])
        )
        if n_fixed:
            transform.SetFixedParameters(values[3 : 3 + n_fixed].tolist())
        transform.SetParameters(values[3 + n_fixed :].tolist())
    except RuntimeError as exc:
        raise ValueError(message) from exc
    return transform


@register_operation(
    category="transform",
    display_name="Compute Registration",
    array_params=("Fixed_Image",),
    recordable_toggle=False,
)
def compute_registration(
    imarray: np.ndarray,
    Fixed_Image: np.ndarray,
    Transform_Type: Literal["Affine", "Rigid", "Similarity", "Translation"] = "Rigid",
    Metric: Literal[
        "Correlation", "Mean Squares", "Mutual Information"
    ] = "Mutual Information",
    Iterations: Annotated[int, {"min": 1, "max": 10000}] = 200,
) -> np.ndarray:
    """
    Find the transform that aligns an image to a fixed image. The result is a small
    array holding the transform, which Apply Transform uses. Convenience wrapper around
    SimpleITK's registration package:
    https://simpleitk.readthedocs.io/en/master/registrationOverview.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to align.
    Fixed_Image: np.ndarray
        The image to align the first image to.
    Transform_Type: Literal
        The kind of transform to fit. Default is "Rigid" which works best for CT data.
        - "Affine": Allows rotation, translation, anisotropic scaling, and shearing.
        - "Rigid": Allows rotation and translation.
        - "Similarity": Allows rotation, translation, and uniform scaling.
        - "Translation": Only allows translation.
    Metric: Literal
        The similarity metric to compare intensities between images. Default is "Mutual
        Information" and this is the only cross-modal option (e.g., CT-MRI).
        - "Correlation": Pearson correlation, intensities can vary by linear offset.
        - "Mean Squares": Average squared intensity difference, intensities must be in
        similar range to work.
        - "Mutual Information": Images from different modalities with different
        intensities and best general use.
    Iterations: int
        The maximum number of optimizer iterations per resolution level.

    Returns
    -------
    transform: np.ndarray
        The transform, stored as a 1 x N array.
    """

    if util.isrgb(Fixed_Image) or util.isrgb(imarray):
        raise ValueError("One or both images are RGB but both must be grayscale!")
    if Fixed_Image.shape != imarray.shape:
        raise ValueError("Both images must have the same shape!")

    import SimpleITK as sitk

    fixed: sitk.Image = sitk.Cast(
        sitk.GetImageFromArray(np.asarray(Fixed_Image)), sitk.sitkFloat32
    )
    moving_image: sitk.Image = sitk.Cast(
        sitk.GetImageFromArray(np.asarray(imarray)), sitk.sitkFloat32
    )

    initial: sitkTransform = _make_transform(Transform_Type, fixed.GetDimension())
    if Transform_Type != "Translation":
        initial = sitk.CenteredTransformInitializer(
            fixed,
            moving_image,
            initial,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    method: sitk.ImageRegistrationMethod = sitk.ImageRegistrationMethod()
    if Metric == "Correlation":
        method.SetMetricAsCorrelation()
    elif Metric == "Mutual Information":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        method.SetMetricAsMeanSquares()
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.1, 1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=Iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    return _pack(Transform_Type, method.Execute(fixed, moving_image))


@register_operation(
    category="transform",
    display_name="Apply Registration Transform",
    array_params=("Transform",),
)
def apply_transform(
    imarray: np.ndarray,
    Transform: np.ndarray,
    Interpolation: Literal["BSpline", "Linear", "Nearest"] = "Linear",
    Default_Value: float = 0.0,
) -> np.ndarray:
    """
    Apply a transform from Compute Registration to an image. Convenience wrapper around
    SimpleITK's registration package:
    https://simpleitk.readthedocs.io/en/master/registrationOverview.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to transform.
    Transform: np.ndarray
        The transform layer made by Compute Registration.
    Interpolation: Literal
        How to sample the image. Default is "Linear".
        - "BSpline": Spline interpolation (preserves peaks, smoother, slowest).
        - "Linear": Linear interpolation (preserves range, medium speed).
        - "Nearest": No interpolation (uses nearest value, fastest).
    Default_Value: float
        The value used outside the original image.

    Returns
    -------
    registered: np.ndarray
        The transformed image.
    """

    if util.isrgb(imarray):
        raise ValueError("RGB images are not supported!")

    import SimpleITK as sitk

    transform: sitkTransform = _unpack(Transform)
    source: sitk.Image = sitk.GetImageFromArray(np.asarray(imarray))
    if source.GetDimension() != transform.GetDimension():
        raise ValueError("The transform and image must have the same dimensionality!")

    resampled: sitk.Image = sitk.Resample(
        source,
        source,
        transform,
        getattr(sitk, _INTERPOLATORS[Interpolation]),
        Default_Value,
        source.GetPixelID(),
    )

    return sitk.GetArrayFromImage(resampled)
