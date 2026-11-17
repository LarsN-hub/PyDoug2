"""
Image registration.
"""

from typing import Annotated, Literal

import numpy as np
from SimpleITK import Image as sitkImage
from SimpleITK import Transform as sitkTransform

from pydoug_core import util
from pydoug_core.registry import register_operation

_KINDS = ("Affine", "Rigid", "Similarity", "Translation")
_INTERPOLATORS = {
    "Linear": "sitkLinear",
    "Nearest": "sitkNearestNeighbor",
    "BSpline": "sitkBSpline",
}
_QUALITY = {
    "Fast": {
        "shrink": [2, 1],
        "sigmas": [1, 0],
        "iterations": 100,
        "sampling": [0.2, 0.1],
    },
    "Standard": {
        "shrink": [4, 2, 1],
        "sigmas": [2, 1, 0],
        "iterations": 200,
        "sampling": [0.5, 0.25, 0.1],
    },
    "Thorough": {
        "shrink": [8, 4, 2, 1],
        "sigmas": [3, 2, 1, 0],
        "iterations": 500,
        "sampling": [1.0, 0.5, 0.25, 0.1],
    },
}


def _to_sitk(
    imarray: np.ndarray,
    spacing: tuple = (1, 1, 1),
) -> sitkImage:
    """
    Convert a numpy array (z, y, x) to SimpleITK image with physical spacing (x, y, z).

    Parameters
    ----------
    imarray: np.ndarray
        The numpy array to convert.
    spacing: tuple[float, float, float] | None
        The spacing of the image along each axis.

    Returns
    -------
    img: sitkImage
        The Simple ITK image object.
    """

    from SimpleITK import GetImageFromArray

    img: sitkImage = GetImageFromArray(np.asarray(imarray))
    if util.is3d(imarray):
        space_vals: tuple[float, float, float] | None = (
            None if spacing[0] == spacing[1] == spacing[2] == 0 else spacing
        )
    else:
        space_vals: tuple[float, float, float] | None = (
            None if spacing[0] == spacing[1] == 0 else spacing
        )
    if space_vals is not None:
        img.SetSpacing(tuple(float(s) for s in space_vals[::-1]))

    return img


def _robust_float(imarray: np.ndarray) -> np.ndarray:
    """
    Clip outliers and rescale to [0, 1] so a few bright pixels don't dominate.

    Parameters
    ----------
    imarray: np.ndarray
        The image to conver to float.

    Returns
    -------
    float_im: np.ndarray
        The image converted to float.
    """

    arr: np.ndarray = np.asarray(imarray, dtype=np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr)

    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _make_transform(
    kind: Literal["Affine", "Rigid", "Similarity", "Translation"] = "Rigid",
    dim: int = 2,
) -> sitkTransform:
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

    from SimpleITK import (
        AffineTransform,
        Euler2DTransform,
        Euler3DTransform,
        Similarity2DTransform,
        Similarity3DTransform,
        TranslationTransform,
    )

    if kind == "Affine":
        return AffineTransform(dim)
    if kind == "Rigid":
        return Euler2DTransform() if dim == 2 else Euler3DTransform()
    if kind == "Similarity":
        return Similarity2DTransform() if dim == 2 else Similarity3DTransform()
    return TranslationTransform(dim)


def _pack(
    kind: Literal["Affine", "Rigid", "Similarity", "Translation"],
    transform: sitkTransform,
    ref_size: list[int],
    ref_spacing: list[float],
) -> np.ndarray:
    """
    Return a numpy array with the kind index, dimension, center size, center, and
    parameters of the transform.

    Parameters
    ----------
    kind: str
        The kind of transform being performed.
    transform: Transform
        The SimpleITK transform.
    ref_size: list[int]
        The size of each axis.
    ref_spacing: list[float]
        The spacing between axes.

    Returns
    -------
    pack_array: np.ndarray
        The numpy array.
    """

    fixed: list = list(transform.GetFixedParameters())
    params: list = list(transform.GetParameters())
    dim: int = transform.GetDimension()
    header: list = [_KINDS.index(kind), dim, *ref_size, *ref_spacing, len(fixed)]

    return np.array([header + fixed + params], dtype=float)


def _unpack(vector: np.ndarray) -> tuple[sitkTransform, list[int], list[float]]:
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

    v: np.ndarray = np.asarray(vector, dtype=float).ravel()
    msg: str = "The Transform layer was not made by Compute Registration."

    if v.size < 3 or not 0 <= v[0] < len(_KINDS) or v[1] not in (2, 3):
        raise ValueError(msg)

    try:
        kind_idx, dim = int(v[0]), int(v[1])  # transformation kind and dimensionality
        i: int = 2
        ref_size: list[int] = [int(x) for x in v[i : i + dim]]  # axis sizing
        i += dim
        ref_spacing: list[float] = [float(x) for x in v[i : i + dim]]  # axis spacing
        i += dim
        n_fixed: int = int(v[i])  # number of fixed parameters
        i += 1
        transform: sitkTransform = _make_transform(_KINDS[kind_idx], dim)
        if n_fixed:
            transform.SetFixedParameters(
                v[i : i + n_fixed].tolist()
            )  # fixed parameters
        i += n_fixed
        transform.SetParameters(v[i:].tolist())  # other parameters

    except (IndexError, RuntimeError) as exc:
        raise ValueError(msg) from exc

    return transform, ref_size, ref_spacing


@register_operation(
    category="register",
    display_name="Compute Registration",
    array_params=("Fixed_Image",),
    recordable_toggle=True,
)
def compute_registration(
    imarray: np.ndarray,
    Fixed_Image: np.ndarray,
    Transform_Type: Literal["Rigid", "Similarity", "Affine", "Translation"] = "Rigid",
    Metric: Literal[
        "Mutual Information", "Correlation", "Mean Squares"
    ] = "Mutual Information",
    Quality: Literal["Fast", "Standard", "Thorough"] = "Standard",
    Moving_Spacing: tuple[
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
    ] = (1, 1, 1),
    Fixed_Spacing: tuple[
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
    ] = (1, 1, 1),
) -> np.ndarray:  # pragma: no cover
    """
    Find the transform that maps the fixed image's space onto the moving image.
    Returns a 1 x N array that Apply Registration Transform uses.
    Use the widgets in the transform tab to line the images up somewhat beforehand for
    better performance.

    Parameters
    ----------
    imarray: np.ndarray
        The image to align (moving).
    Fixed_Image: np.ndarray
        The reference image. Its grid is the output grid when applying the transform.
        May differ in shape from imarray.
    Transform_Type: Literal
        - "Translation": Shift only.
        - "Rigid": Rotation and shift. Good default for same sample CT scans.
        - "Similarity": Rigid plus uniform scale.
        - "Affine": Adds anisotropic scale and shear. Use only when needed.
    Metric: Literal
        - "Mutual Information": Default, can do different modalities (CT to MRI).
        - "Correlation": Same modality, possibly with a brightness or contrast offset.
        - "Mean Squares": Same modality and same intensity scale.
    Quality: Literal
        Controls the pyramid levels, iterations, and sampling.
        - "Fast": Quick check or large data.
        - "Standard": Most cases.
        - "Thorough": Hard cases, small rotations, or low overlap.
    Moving_Spacing: tuple[float, float, float]
        Voxel size in array order (x, y, z). Z-axis ignored if image is 2D.
    Fixed_Spacing: tuple[float, float, float]
        Voxel size in array order (x, y, z). Z-axis ignored if image is 2D.

    Returns
    -------
    transform: np.ndarray
        The transform, stored as a 1 x N array.
    """

    if util.isrgb(Fixed_Image) or util.isrgb(imarray):
        raise ValueError("Both images must be grayscale.")
    if imarray.ndim != Fixed_Image.ndim or imarray.ndim not in (2, 3):
        raise ValueError("Both images must be 2D/3D with the same dimensionality.")

    import SimpleITK as sitk

    q: dict[str, int | list[float]] = _QUALITY[Quality]
    fix_space: tuple = (
        (Fixed_Spacing[2], Fixed_Spacing[1], Fixed_Spacing[0])
        if util.is3d(imarray)
        else (Fixed_Spacing[1], Fixed_Spacing[0])
    )
    move_space: tuple = (
        (Moving_Spacing[2], Moving_Spacing[1], Moving_Spacing[0])
        if util.is3d(imarray)
        else (Moving_Spacing[1], Moving_Spacing[0])
    )
    fixed: sitkImage = _to_sitk(_robust_float(Fixed_Image), fix_space)
    moving: sitkImage = _to_sitk(_robust_float(imarray), move_space)

    initial: sitkTransform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        _make_transform(Transform_Type, fixed.GetDimension()),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    method: sitk.ImageRegistrationMethod = sitk.ImageRegistrationMethod()
    if Metric == "Mutual Information":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    elif Metric == "Correlation":
        method.SetMetricAsCorrelation()
    else:
        method.SetMetricAsMeanSquares()

    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentagePerLevel(q["sampling"], seed=1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=q["iterations"],
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(q["shrink"])
    method.SetSmoothingSigmasPerLevel(q["sigmas"])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    final: sitkTransform = method.Execute(fixed, moving)

    return _pack(
        Transform_Type,
        final,
        ref_size=list(fixed.GetSize()),
        ref_spacing=list(fixed.GetSpacing()),
    )


@register_operation(
    category="register",
    display_name="Apply Registration Transform",
    array_params=("Transform",),
)
def apply_transform(
    imarray: np.ndarray,
    Transform: np.ndarray,
    Interpolation: Literal["Linear", "Nearest", "BSpline"] = "Linear",
    Default_Value: float = 0.0,
    Moving_Spacing: tuple[
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 100000000000000, "step": 0.0001}],
    ] = (1, 1, 1),
) -> np.ndarray:  # pragma: no cover
    """
    Resample an image into the fixed image's space using a transform from
    Compute Registration. The output has the same shape and spacing as the
    fixed image that was used to compute the transform.

    Parameters
    ----------
    imarray: np.ndarray
        The image to transform (moving image). Raw intensities are preserved.
    Transform: np.ndarray
        The transform layer made by Compute Registration.
    Interpolation: Literal
        - "Linear": general use for intensity images.
        - "Nearest": use for label or mask images so no new label values appear.
        - "BSpline": smoothest result, slowest.
    Default_Value: float
        The value used outside the original image.
    Moving_Spacing: tuple[float, float, float]
        Voxel size of imarray in array order, same as used for Compute Registration.

    Returns
    -------
    registered: np.ndarray
        The transformed image, on the fixed image's grid.
    """

    if util.isrgb(imarray):
        raise ValueError("RGB images are not supported.")

    import SimpleITK as sitk

    transform, ref_size, ref_spacing = _unpack(Transform)
    move_space: tuple = (
        (Moving_Spacing[2], Moving_Spacing[1], Moving_Spacing[0])
        if util.is3d(imarray)
        else (Moving_Spacing[1], Moving_Spacing[0])
    )
    moving: sitkImage = _to_sitk(imarray, move_space)
    if moving.GetDimension() != transform.GetDimension():
        raise ValueError("The transform and image must have the same dimensionality.")

    reference: sitkImage = sitk.Image(ref_size, moving.GetPixelID())
    reference.SetSpacing(ref_spacing)
    resampled: sitkImage = sitk.Resample(
        moving,
        reference,
        transform,
        getattr(sitk, _INTERPOLATORS[Interpolation]),
        Default_Value,
        moving.GetPixelID(),
    )

    return sitk.GetArrayFromImage(resampled)
