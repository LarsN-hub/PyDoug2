"""
Tests for PyDoug's results module.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from pydoug_core import results


def test_PlotResult(tmp_path) -> None:

    data: dict[str, Sequence] = {
        "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    plot_result: results.PlotResult = results.PlotResult(Figure(), data)
    plot_result.write_csv(Path(tmp_path / "test_write_csv.csv"))
    assert Path(tmp_path / "test_write_csv.csv").exists()


def test_PlotResult_error() -> None:

    data: dict[str, Sequence] = {
        "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "2": [1, 2, 3, 4, 5],
        "3": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    with pytest.raises(ValueError):
        _: results.PlotResult = results.PlotResult(Figure(), data)
