"""
Tests for PyDoug Core's batch recorder module.
"""

from pathlib import Path

import numpy as np
import pytest

import pydoug_core  # noqa: F401 — imported for its registration side effect
from pydoug_core.batch import recorder
from pydoug_core.mask.logic import apply_mask
from pydoug_core.trans.geom import mirror, reslice


def test_Recorder(tmp_path, sample_3dgray_array, sample_masks) -> None:

    class DummyLayer:
        pass

    layer_ref1: DummyLayer = DummyLayer()
    record1: recorder.Recorder = recorder.Recorder()
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref1)
    record1.record(
        "apply_mask",
        {},
        input_refs={
            "layer": f"step:{record1.steps[0].step_id}",
            "Mask": "pending:anything",
        },
        pending_masks={"Mask": sample_masks["3dgray"]},
    )
    record1.record(
        "reslice",
        {"Orientation": "Left"},
        input_refs={"layer": f"step:{record1.steps[1].step_id}"},
    )
    assert record1.step_id_for_layer(layer_ref1) == record1.steps[0].step_id
    assert record1.dependents_of(record1.steps[0].step_id) == record1.steps[1:]
    result0: np.ndarray = record1.replay(sample_3dgray_array)
    record1.save(Path(tmp_path / "test_params.json"))
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_params.json"))
    result1: np.ndarray = record1.replay(sample_3dgray_array)
    result2: np.ndarray = record2.replay(sample_3dgray_array)
    result3: np.ndarray = reslice(
        apply_mask(mirror(sample_3dgray_array), sample_masks["3dgray"]), "Left"
    )
    for index, item in enumerate(record1.steps):
        assert item.step_id == record2.steps[index].step_id
    assert np.all(result0 == result1)
    assert np.all(result1 == result2)
    assert result1.shape != sample_3dgray_array.shape
    assert np.all(result1 == result3)
    layer_ref0: DummyLayer = DummyLayer()
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref0)
    record1.remove_step(record1.steps[-1].step_id)
    assert record1.steps[-1].step_id == record2.steps[-1].step_id
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref0)
    record1.remove_steps_by_layer(record1.steps[-1].layer_ref)
    assert record1.steps[-1].step_id == record2.steps[-1].step_id
    record1.clear()
    assert record1.steps == []
    save_workflow_dir: Path | None = record2._workflow_dir
    record2._workflow_dir = None
    with pytest.raises(ValueError):
        record2.replay(sample_3dgray_array)
    record2._workflow_dir = save_workflow_dir
    record2.steps[1].input_refs["Mask"] = "test"
    with pytest.raises(ValueError):
        record2.replay(sample_3dgray_array)
