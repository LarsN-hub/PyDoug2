"""
Tests for PyDoug's batch._result_save module.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from pydoug_core.batch import _result_save
from pydoug_core.results import PlotResult


def test_default_save_result(tmp_path) -> None:

    result1: Figure = Figure()
    result2: dict[str, float | int | str] = {"test1": 1, "test2": 1.5, "test3": "test"}
    result3: int = 2
    result4: dict[str, np.float64] = {"test1": np.float64(1.2)}
    result5: tuple = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), np.array([7]))
    result6: PlotResult = PlotResult(Figure(), {"test": [1, 2, 3, 4, 5]})
    path1: Path = Path(tmp_path / "save_fig_test")
    path2: Path = Path(tmp_path / "save_dict_test")
    path3: Path = Path(tmp_path / "save_int_test")
    path4: Path = Path(tmp_path / "save_npdict_test")
    path5: Path = Path(tmp_path / "save_tuple_test")
    path6: Path = Path(tmp_path / "save_PlotResult_test")
    _result_save.default_save_result(result1, path1)
    _result_save.default_save_result(result2, path2)
    _result_save.default_save_result(result3, path3)
    _result_save.default_save_result(result4, path4)
    _result_save.default_save_result(result5, path5)
    _result_save.default_save_result(result6, path6)
    assert Path(tmp_path / "save_fig_test.png").exists()
    assert Path(tmp_path / "save_dict_test.json").exists()
    assert Path(tmp_path / "save_int_test.txt").exists()
    assert Path(tmp_path / "save_npdict_test.json").exists()
    assert Path(tmp_path / "save_tuple_test.npz").exists()
    assert Path(tmp_path / "save_PlotResult_test.csv").exists()
    assert Path(tmp_path / "save_PlotResult_test.png").exists()
