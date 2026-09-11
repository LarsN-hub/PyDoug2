"""
Shared fixtures for pydoug-napari tests.
"""

import numpy as np
import pytest

pytest_plugins: list[str] = ["pydoug_core.testing"]


@pytest.fixture
def registered_half():
    from pydoug_core.registry import register_operation, unregister_operation

    @register_operation(category="testing", display_name="Half")
    def _half(imarray: np.ndarray, factor: float = 0.5) -> np.ndarray:
        return np.round(imarray * factor).astype(imarray.dtype)  # pragma: no cover

    yield "_half"
    unregister_operation("_half")


@pytest.fixture
def registered_mask():
    from pydoug_core.registry import register_operation, unregister_operation

    @register_operation(
        category="testing", display_name="Mask", mask_params=("maskarray",)
    )
    def _mask(imarray: np.ndarray, maskarray: np.ndarray) -> np.ndarray:
        output: np.ndarray = np.copy(imarray)  # pragma: no cover
        output[np.logical_not(maskarray)] = 0  # pragma: no cover
        return output  # pragma: no cover

    yield "_mask"
    unregister_operation("_mask")


@pytest.fixture
def registered_bool():
    from pydoug_core.registry import register_operation, unregister_operation

    @register_operation(
        category="testing", display_name="Mask", mask_params=("maskarray",)
    )
    def _bool(imarray: np.ndarray, thresh: int = 128) -> np.ndarray:
        output: np.ndarray = np.copy(imarray)  # pragma: no cover
        output[output > thresh] = 255  # pragma: no cover
        output[output <= thresh] = 0  # pragma: no cover
        return np.astype(output, np.bool)  # pragma: no cover

    yield "_mask"
    unregister_operation("_mask")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
