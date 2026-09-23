"""
Reusable pytest fixtures for testing code that consumes pydoug-core arrays.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_2dgray_array() -> np.ndarray:
    """
    Fixture to create a sample 2d grayscale image array.
    """

    return np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)


@pytest.fixture
def sample_3dgray_array() -> np.ndarray:
    """
    Fixture to create a sample 3d grayscale image array.
    """

    return np.random.randint(0, 256, size=(5, 10, 10), dtype=np.uint8)


@pytest.fixture
def sample_2drgb_array() -> np.ndarray:
    """
    Fixture to create a sample 2d RGB image array.
    """

    return np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)


@pytest.fixture
def sample_3drgb_array() -> np.ndarray:
    """
    Fixture to create a sample 2d RGB image array.
    """

    return np.random.randint(0, 256, size=(5, 10, 10, 3), dtype=np.uint8)


@pytest.fixture
def sample_arrays(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> dict[str, np.ndarray]:
    """
    Registry for mapping kind-name to corresponding sample image array.
    """

    return {
        "2dgray": sample_2dgray_array,
        "3dgray": sample_3dgray_array,
        "2drgb": sample_2drgb_array,
        "3drgb": sample_3drgb_array,
    }


@pytest.fixture
def sample_masks(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> dict[str, np.ndarray]:
    """
    Registry for mapping kind-name to corresponding sample mask array.
    """

    mask_2dgray: np.ndarray = np.copy(sample_2dgray_array)
    mask_2dgray[mask_2dgray < 128] = 0
    mask_2dgray[mask_2dgray >= 128] = 255
    mask_2dgray = np.astype(mask_2dgray, np.bool)
    mask_3dgray: np.ndarray = np.copy(sample_3dgray_array)
    mask_3dgray[mask_3dgray < 128] = 0
    mask_3dgray[mask_3dgray >= 128] = 255
    mask_3dgray = np.astype(mask_3dgray, np.bool)
    mask_2drgb: np.ndarray = np.copy(sample_2drgb_array)
    mask_2drgb[mask_2drgb < 128] = 0
    mask_2drgb[mask_2drgb >= 128] = 255
    mask_2drgb = np.astype(mask_2drgb, np.bool)
    mask_3drgb: np.ndarray = np.copy(sample_3drgb_array)
    mask_3drgb[mask_3drgb < 128] = 0
    mask_3drgb[mask_3drgb >= 128] = 255
    mask_3drgb = np.astype(mask_3drgb, np.bool)

    return {
        "2dgray": mask_2dgray,
        "3dgray": mask_3dgray,
        "2drgb": mask_2drgb,
        "3drgb": mask_3drgb,
    }
