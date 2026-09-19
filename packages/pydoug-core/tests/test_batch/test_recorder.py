"""
Tests for PyDoug Core's batch recorder module.
"""

from pathlib import Path
from shutil import rmtree
from typing import Any

import numpy as np
import pytest

import pydoug_core  # noqa: F401 — imported for its registration side effect
from pydoug_core.batch import recorder
from pydoug_core.io import imwriter
from pydoug_core.mask.logic import apply_mask
from pydoug_core.trans.geom import mirror, reslice


class DummyLayer:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data = data


def test_Recorder(tmp_path, sample_3dgray_array, sample_masks) -> None:

    layer_ref1: DummyLayer = DummyLayer(None)
    mask_layer_ref: DummyLayer = DummyLayer(sample_masks["3dgray"])
    record1: recorder.Recorder = recorder.Recorder()
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref1)
    record1.record(
        "apply_mask",
        {},
        input_refs={
            "layer": f"step:{record1.steps[0].step_id}",
            "Mask": record1.get_or_create_pending_ref(mask_layer_ref),
        },
    )
    record1.record(
        "apply_mask",
        {},
        input_refs={
            "layer": f"step:{record1.steps[0].step_id}",
            "Mask": record1.get_or_create_pending_ref(mask_layer_ref),
        },
    )
    record1.record(
        "reslice",
        {"Orientation": "Left"},
        input_refs={"layer": f"step:{record1.steps[1].step_id}"},
    )
    mask_save_path: Path = Path(
        tmp_path / f"test_params/arrays/{list(record1._pending_arrays.keys())[0]}.tiff"
    )
    assert record1.step_id_for_layer(layer_ref1) == record1.steps[0].step_id
    assert record1.dependents_of(record1.steps[0].step_id) == record1.steps[1:]
    result1: np.ndarray | dict | Any = record1.replay(sample_3dgray_array)
    assert isinstance(result1, np.ndarray)
    record1.save(Path(tmp_path / "test_params"))
    record1.save(Path(tmp_path / "test_params_duplicate"))
    assert Path(tmp_path / "test_params").exists()
    assert Path(tmp_path / "test_params_duplicate").exists()
    assert Path(tmp_path / "test_params/arrays").exists()
    assert mask_save_path.exists()
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_params"))
    with pytest.raises(RuntimeError):
        rmtree(Path(tmp_path / "test_params"))
        rmtree(Path(tmp_path / "test_params_duplicate"))
        record1.save(Path(tmp_path / "test_params_duplicate_2"))
    result2: np.ndarray = reslice(
        apply_mask(mirror(sample_3dgray_array), sample_masks["3dgray"]), "Left"
    )
    assert result1.shape != sample_3dgray_array.shape
    assert np.all(result1 == result2)
    layer_ref2: DummyLayer = DummyLayer(None)
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref2)
    rem_step: Any | str = record1.steps[-1].layer_ref
    record1.remove_step(record1.steps[-1].step_id)
    assert record1.steps[-1].step_id != rem_step
    record1.record("mirror", {}, input_refs={"layer": "primary"}, layer_ref=layer_ref2)
    rem_step = record1.steps[-1].step_id
    record1.remove_steps_by_layer(record1.steps[-1].layer_ref)
    assert record1.steps[-1].step_id != rem_step
    record1.clear()
    assert record1.steps == []


def test_runtime_error(sample_masks, tmp_path, monkeypatch) -> None:

    layer_ref: DummyLayer = DummyLayer(None)
    mask_layer_ref: DummyLayer = DummyLayer(sample_masks["3dgray"])
    record1: recorder.Recorder = recorder.Recorder()
    record1.record(
        "apply_mask",
        {},
        input_refs={
            "layer": "primary",
            "Mask": record1.get_or_create_pending_ref(mask_layer_ref),
        },
        layer_ref=layer_ref,
    )
    with pytest.raises(RuntimeError):

        def _skip(one, two, three, four) -> None:
            return

        monkeypatch.setattr(imwriter, "imwrite", _skip)
        record1.save(Path(tmp_path / "test_params_fail"))
