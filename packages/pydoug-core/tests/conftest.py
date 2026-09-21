"""
Shared fixtures for pydoug-core tests.
"""

import numpy as np
import pytest

from pydoug_core.registry import register_operation, unregister_operation

pytest_plugins: list[str] = ["pydoug_core.testing"]


@pytest.fixture
def registered_nothing():
    """
    Fixture to create a registered "nothing" operation.
    """

    @register_operation(category="testing", display_name="Nothing")
    def _nothing(
        imarray: np.ndarray,
        extra_array: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.copy(imarray)  # pragma: no cover

    yield "_nothing"
    unregister_operation("_nothing")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
