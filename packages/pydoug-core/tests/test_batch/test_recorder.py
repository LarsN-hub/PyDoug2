"""
Tests for PyDoug Core's batch recorder module.
"""

from pathlib import Path

import numpy as np

from pydoug_core.batch import recorder
from pydoug_core.trans.geom import mirror, reslice


def test_recorder(tmp_path, sample_3dgray_array) -> None:

    result1: recorder.Recorder = recorder.Recorder()
    result1.record("mirror", {})
    result1.record("reslice", {"orientation": "left"})
    result1.save(Path(tmp_path / "test_params.json"))
    result2: recorder.Recorder = recorder.Recorder()
    result2.load(Path(tmp_path / "test_params.json"))
    result3: np.ndarray = result1.replay(sample_3dgray_array)
    result4: np.ndarray = result2.replay(sample_3dgray_array)
    assert result1.steps == result2.steps
    assert np.all(result3 == result4)
    assert result3.shape != sample_3dgray_array.shape
    assert np.all(result3 == reslice(mirror(sample_3dgray_array), "left"))
