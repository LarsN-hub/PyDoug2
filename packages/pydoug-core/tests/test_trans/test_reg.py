"""
Tests for PyDoug's trans.reg module.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from pydoug_core.trans import reg


def test_to_sitk(sample_arrays) -> None:

    result01: sitk.Image = reg._to_sitk(sample_arrays["2dgray"])
    result02: sitk.Image = reg._to_sitk(sample_arrays["3dgray"])
    result03: sitk.Image = reg._to_sitk(sample_arrays["2dgray"], (0, 0, 0))
    result04: sitk.Image = reg._to_sitk(sample_arrays["3dgray"], (0, 0, 0))
    assert isinstance(result01, sitk.Image)
    assert isinstance(result02, sitk.Image)
    assert isinstance(result03, sitk.Image)
    assert isinstance(result04, sitk.Image)


def test_robust_float(sample_arrays) -> None:

    result00: np.ndarray = reg._robust_float(np.ones((10, 10), np.uint8) * 128)
    result01: np.ndarray = reg._robust_float(sample_arrays["2dgray"])
    result02: np.ndarray = reg._robust_float(sample_arrays["3dgray"])
    result03: np.ndarray = reg._robust_float(sample_arrays["2dgray"])
    result04: np.ndarray = reg._robust_float(sample_arrays["3dgray"])
    assert isinstance(result00, np.ndarray)
    assert isinstance(result01, np.ndarray)
    assert isinstance(result02, np.ndarray)
    assert isinstance(result03, np.ndarray)
    assert isinstance(result04, np.ndarray)
    assert result01.dtype == np.float64
    assert result02.dtype == np.float64
    assert result03.dtype == np.float64
    assert result04.dtype == np.float64


def test_make_transform() -> None:

    result01: sitk.Transform = reg._make_transform()
    result02: sitk.Transform = reg._make_transform(dim=3)
    result03: sitk.Transform = reg._make_transform("Affine")
    result04: sitk.Transform = reg._make_transform("Similarity")
    result05: sitk.Transform = reg._make_transform("Similarity", 3)
    result06: sitk.Transform = reg._make_transform("Translation")
    assert isinstance(result01, sitk.Euler2DTransform)
    assert isinstance(result02, sitk.Euler3DTransform)
    assert isinstance(result03, sitk.AffineTransform)
    assert isinstance(result04, sitk.Similarity2DTransform)
    assert isinstance(result05, sitk.Similarity3DTransform)
    assert isinstance(result06, sitk.TranslationTransform)


def test_pack() -> None:

    result01: np.ndarray = reg._pack(
        "Affine", sitk.AffineTransform(2), [10, 10], [1, 1]
    )
    result02: np.ndarray = reg._pack("Rigid", sitk.Euler2DTransform(), [10, 10], [1, 1])
    result03: np.ndarray = reg._pack(
        "Rigid", sitk.Euler3DTransform(), [10, 10, 10], [1, 1, 1]
    )
    result04: np.ndarray = reg._pack(
        "Similarity", sitk.Similarity2DTransform(), [10, 10], [1, 1]
    )
    result05: np.ndarray = reg._pack(
        "Similarity", sitk.Similarity3DTransform(), [10, 10, 10], [1, 1, 1]
    )
    result06: np.ndarray = reg._pack(
        "Translation", sitk.TranslationTransform(2), [10, 10], [1, 1]
    )
    assert isinstance(result01, np.ndarray)
    assert isinstance(result02, np.ndarray)
    assert isinstance(result03, np.ndarray)
    assert isinstance(result04, np.ndarray)
    assert isinstance(result05, np.ndarray)
    assert isinstance(result06, np.ndarray)


def test_unpack() -> None:

    with pytest.raises(ValueError):
        _ = reg._unpack(np.array([1, 2]))
    with pytest.raises(ValueError):
        _ = reg._unpack(np.array([-1, 2, 3]))
    with pytest.raises(ValueError):
        _ = reg._unpack(np.array([1, 2, 10, 10, 1, 1, 2, 1, 1]))

    result01: tuple[sitk.Transform, list[int], list[float]] = reg._unpack(
        np.array([1, 2, 10, 10, 1, 1, 2, 1, 1, 1, 1, 1])
    )
    assert isinstance(result01[0], sitk.Euler2DTransform)
    assert isinstance(result01[1], list)
    assert isinstance(result01[2], list)
