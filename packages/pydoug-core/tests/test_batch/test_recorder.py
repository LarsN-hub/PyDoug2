"""
Tests for PyDoug Core's batch recorder module.
"""

from pathlib import Path

import numpy as np

from pydoug_core.batch import recorder
from pydoug_core.trans.geom import mirror, reslice


def test_Recorder_record_save_load_remove_clear(tmp_path, sample_3dgray_array) -> None:

    record1: recorder.Recorder = recorder.Recorder()
    record1.record("mirror", {}, input_refs={"layer": "primary"})
    record1.record(
        "reslice",
        {"Orientation": "Left"},
        input_refs={"layer": f"step:{record1.steps[0].step_id}"},
    )
    print(record1.steps)
    record1.save(Path(tmp_path / "test_params.json"))
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_params.json"))
    result1: np.ndarray = record1.replay(sample_3dgray_array)
    result2: np.ndarray = record2.replay(sample_3dgray_array)
    result3: np.ndarray = reslice(mirror(sample_3dgray_array), "Left")
    assert record1.steps == record2.steps
    assert np.all(result1 == result2)
    assert result1.shape != sample_3dgray_array.shape
    assert np.all(
        result1 == result3
    )  # each step calls first array I think, need to call subsequent
    layer_ref1: object = object()
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref1)
    record1.remove_steps_by_layer(layer_ref1)
    assert record1.steps == record2.steps
    record1.clear()
    assert record1.steps == []
