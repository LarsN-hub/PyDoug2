"""
Shared fixtures for pydoug-core tests.
"""

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def sample_2dgray_array() -> np.ndarray:
    """
    Fixture to create a sample 2d grayscale image array.
    """

    return np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)


@pytest.fixture
def sample_3dgray_array() -> np.ndarray:
    """
    Fixture to create a sample 3d grayscale image array.
    """

    return np.random.randint(0, 255, size=(5, 10, 10), dtype=np.uint8)


@pytest.fixture
def sample_2drgb_array() -> np.ndarray:
    """
    Fixture to create a sample 2d RGB image array.
    """

    return np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8)


@pytest.fixture
def sample_3drgb_array() -> np.ndarray:
    """
    Fixture to create a sample 2d RGB image array.
    """

    return np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)


@pytest.fixture
def sample_arrays(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> dict[str, np.ndarray]:
    """
    Registry for mapping kind-name to corresponding sample array.
    """

    return {
        "2dgray": sample_2dgray_array,
        "3dgray": sample_3dgray_array,
        "2drgb": sample_2drgb_array,
        "3drgb": sample_3drgb_array,
    }


@pytest.fixture
def make_array() -> Callable:
    """
    Fixture that returns a function for making an image array with a specified shape.
    """

    def _make(shape=(10, 10), dtype=np.uint8):
        if issubclass(dtype, np.floating):
            minval = np.finfo(dtype).min
            maxval = np.finfo(dtype).max
        elif issubclass(dtype, np.integer):
            minval = np.iinfo(dtype).min
            maxval = np.iinfo(dtype).max
        else:
            minval, maxval = 0, 1
        return np.random.randint(minval, maxval, size=shape, dtype=dtype)

    return _make
