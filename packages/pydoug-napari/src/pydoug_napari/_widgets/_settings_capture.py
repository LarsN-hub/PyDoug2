"""
Convert captured viewer settings to json-serializable formats.
"""

from enum import Enum
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """
    Recursively convert napari/numpy objects into plain JSON-safe types.

    Parameters
    ----------
    value: Any
        The value to convert.

    Returns
    -------
    value: Any
        The value in json-serializable format.
    """

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()  # pragma: no cover
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        return to_jsonable(value.dict())  # pragma: no cover
    return value
