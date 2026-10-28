"""
Tests for PyDoug Core's batch recorder module.
"""

from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import pytest

import pydoug_core  # noqa: F401 — imported for its registration side effect
from pydoug_core.batch import recorder
from pydoug_core.io import imreader, imwriter
from pydoug_core.payloads import ShapesData


class DummyLayer:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data = data


class DummySurfaceLayer:
    def __init__(self, data: tuple | None) -> None:
        self.data = data


class DummyShapesLayer:
    def __init__(self, data: ShapesData | None) -> None:
        self.data = data


def test_RecordedStep() -> None:

    recorded_step: recorder.RecordedStep = recorder.RecordedStep(
        "test_id", "test_name", {}, {}, DummyLayer(None)
    )
    assert recorded_step.step_id == "test_id"
    assert recorded_step.operation_name == "test_name"
    assert recorded_step.input_refs == recorded_step.params == {}
    assert isinstance(recorded_step.layer_ref, DummyLayer)


def test_Recorder_init() -> None:

    record: recorder.Recorder = recorder.Recorder()
    assert record.steps == []
    assert isinstance(record._layer_to_step, WeakKeyDictionary)
    assert isinstance(record._external_layer_refs, WeakKeyDictionary)
    assert record._arrays == {}


def test_Recorder_record_and_step_id_for_layer() -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer_ref: DummyLayer = DummyLayer(None)
    record.record("test_name", {}, {}, None)
    record.record("test_name2", {}, {}, layer_ref)
    assert len(record.steps) == 2
    assert record.steps[0].operation_name == "test_name"
    assert record.steps[1].operation_name == "test_name2"
    assert record._layer_to_step[layer_ref] == record.steps[1].step_id
    assert record.steps[1].step_id == record.step_id_for_layer(layer_ref)


def test_Recorder_get_or_create_array_ref(sample_2dgray_array) -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(sample_2dgray_array)
    layer2: DummyLayer = DummyLayer(sample_2dgray_array)
    ref1: str = record.get_or_create_array_ref(layer1)
    ref2: str = record.get_or_create_array_ref(layer1)
    ref3: str = record.get_or_create_array_ref(layer2)
    assert ref1 == ref2
    assert ref1 != ref3


def test_Recorder_register_array(sample_2dgray_array) -> None:

    record: recorder.Recorder = recorder.Recorder()
    ref: str = record.register_array(sample_2dgray_array)
    assert np.all(record._arrays[ref.split(":", 1)[1]] == sample_2dgray_array)


def test_Recorder_remove_steps_and_remove_step_by_layer_and_clear() -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer: DummyLayer = DummyLayer(None)
    record.record("test_name", {}, {}, None)
    record.record("test_name2", {}, {}, layer)
    assert len(record.steps) == 2
    record.remove_step(record.steps[0].step_id)
    assert len(record.steps) == 1
    record.remove_steps_by_layer(layer)
    assert len(record.steps) == 0
    record.record("test_name", {}, {}, None)
    assert len(record.steps) == 1
    record.clear()
    assert len(record.steps) == 0


def test_Recorder_dependents_of() -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(None)
    layer2: DummyLayer = DummyLayer(None)
    record.record("test_name", {}, {"layer": "primary"}, layer1)
    record.record(
        "test_name2", {}, {"layer": f"step:{record.step_id_for_layer(layer1)}"}, layer2
    )
    record.record(
        "test_name3", {}, {"layer": f"step:{record.step_id_for_layer(layer2)}"}, None
    )
    deps1: list[recorder.RecordedStep] = record.dependents_of(record.steps[0].step_id)
    deps2: list[recorder.RecordedStep] = record.dependents_of(record.steps[1].step_id)
    deps3: list[recorder.RecordedStep] = record.dependents_of(record.steps[2].step_id)
    assert len(deps3) == 0
    assert len(deps2) == 1
    assert len(deps1) == 2


def test_Recorder_save_and_load(tmp_path, sample_2dgray_array) -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(None)
    layer2: DummyLayer = DummyLayer(None)
    ref: str = record.register_array(sample_2dgray_array)
    record.record("test_name", {}, {"layer": "primary"}, layer1)
    record.record(
        "test_name2",
        {},
        {"layer": f"step:{record.step_id_for_layer(layer1)}", "array": ref},
        layer2,
    )
    record.save(Path(tmp_path / "test_workflow"))
    assert Path(tmp_path / "test_workflow/workflow.json").exists()
    assert Path(tmp_path / f"test_workflow/arrays/{ref.split(':', 1)[1]}.tiff").exists()
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_workflow"))
    assert len(record2.steps) == len(record.steps)


def test_save_RuntimeError(sample_2dgray_array, tmp_path, monkeypatch) -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(None)
    layer2: DummyLayer = DummyLayer(sample_2dgray_array)
    record.record(
        "test_name",
        {},
        input_refs={
            "layer": "primary",
            "array": record.get_or_create_array_ref(layer2),
        },
        layer_ref=layer1,
    )
    with pytest.raises(RuntimeError):

        def _skip(one, two, three, four) -> None:
            return

        monkeypatch.setattr(imwriter, "imwrite", _skip)
        record.save(Path(tmp_path / "test_workflow_save_fail"))


def test_load_RuntimeError(sample_2dgray_array, tmp_path, monkeypatch) -> None:

    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(None)
    layer2: DummyLayer = DummyLayer(sample_2dgray_array)
    record.record(
        "test_name",
        {},
        input_refs={
            "layer": "primary",
            "array": record.get_or_create_array_ref(layer2),
        },
        layer_ref=layer1,
    )
    record.save(Path(tmp_path / "test_workflow_load_fail"))
    with pytest.raises(RuntimeError):

        def _skip(one) -> None:
            return

        monkeypatch.setattr(imreader, "imread", _skip)
        record2: recorder.Recorder = recorder.Recorder()
        record2.load(Path(tmp_path / "test_workflow_load_fail"))


def test_resolve(sample_2dgray_array) -> None:

    alt_array1: np.ndarray = sample_2dgray_array / 2
    alt_array2: np.ndarray = sample_2dgray_array / 3
    alt_array3: np.ndarray = sample_2dgray_array / 4
    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(alt_array1)
    layer2: DummyLayer = DummyLayer(alt_array2)
    layer3: DummyLayer = DummyLayer(alt_array3)
    record.record("test_name", {}, {"layer": "primary"}, layer1)
    outputs: dict = {record.step_id_for_layer(layer1): alt_array1}
    record.record(
        "test_name2",
        {},
        {
            "layer": f"step:{record.step_id_for_layer(layer1)}",
            "array": record.get_or_create_array_ref(layer3),
        },
        layer2,
    )
    outputs[record.step_id_for_layer(layer2)] = alt_array2
    array_ref: str = record.get_or_create_array_ref(layer3)

    result1: np.ndarray | tuple | ShapesData | None = record._resolve(
        "primary", sample_2dgray_array, outputs
    )
    assert result1 is not None
    assert np.all(result1 == sample_2dgray_array)

    result2: np.ndarray | tuple | ShapesData | None = record._resolve(
        f"step:{record.step_id_for_layer(layer1)}",
        sample_2dgray_array,
        outputs,
    )
    assert result2 is not None
    assert np.all(result2 == alt_array1)

    result3: np.ndarray | tuple | ShapesData | None = record._resolve(
        f"step:{record.step_id_for_layer(layer2)}",
        sample_2dgray_array,
        outputs,
    )
    assert result3 is not None
    assert np.all(result3 == alt_array2)

    result4: np.ndarray | tuple | ShapesData | None = record._resolve(
        array_ref, sample_2dgray_array, outputs
    )
    assert result4 is not None
    assert np.all(result4 == alt_array3)

    with pytest.raises(ValueError):
        _: np.ndarray | tuple | ShapesData | None = record._resolve(
            "fail", sample_2dgray_array, outputs
        )


def test_Recorder_replay_and_iter_replay(
    sample_2dgray_array,
    tmp_path,
    registered_nothing,
    registered_nothing_no_layer,
) -> None:

    alt_array1: np.ndarray = np.copy(sample_2dgray_array)
    alt_array2: np.ndarray = np.copy(sample_2dgray_array)
    alt_array3: np.ndarray = np.copy(sample_2dgray_array)
    alt_array4: np.ndarray = np.copy(sample_2dgray_array)
    record: recorder.Recorder = recorder.Recorder()
    layer1: DummyLayer = DummyLayer(alt_array1)
    layer2: DummyLayer = DummyLayer(alt_array2)
    layer3: DummyLayer = DummyLayer(alt_array3)
    layer4: DummyLayer = DummyLayer(alt_array4)
    record.record("_nothing", {}, {"layer": "primary"}, layer1)
    record.record(
        "_nothing",
        {},
        {
            "layer": f"step:{record.step_id_for_layer(layer1)}",
            "extra_array": record.get_or_create_array_ref(layer2),
        },
        layer3,
    )
    record.record(
        "_nothing_no_layer",
        {},
        {},
        layer4,
    )
    record.save(Path(tmp_path / "test_workflow_replay"))
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_workflow_replay"))
    result: np.ndarray | dict | Any = record2.replay(sample_2dgray_array)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert not np.all(result == sample_2dgray_array)


def test_surface_behavior(sample_masks, registered_nothing_surface, tmp_path) -> None:

    record: recorder.Recorder = recorder.Recorder()
    dummy_tuple: tuple = (np.array([[0, 0, 0]]), np.array([[1, 1, 1]]), np.array([5]))
    dummy_tuple2: tuple = (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), np.array([7]))
    layer1: DummySurfaceLayer = DummySurfaceLayer(dummy_tuple)
    layer2: DummySurfaceLayer = DummySurfaceLayer(dummy_tuple2)
    record.record(
        "_nothing_surface",
        {},
        {"layer": "primary", "dummy_surface": record.get_or_create_array_ref(layer1)},
        layer2,
    )
    record.save(Path(tmp_path / "test_surface_replay"))
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_surface_replay"))
    result: Any = record2.replay(sample_masks["3dgray"])
    assert result is not None
    assert isinstance(result, tuple)


def test_shapes_behavior(
    sample_2dgray_array, registered_nothing_shapes, tmp_path
) -> None:

    record: recorder.Recorder = recorder.Recorder()
    shapes: ShapesData = ShapesData([np.array([[1, 2], [3, 4]])], ["line"])
    layer1: DummyShapesLayer = DummyShapesLayer(shapes)
    layer2: DummyLayer = DummyLayer(sample_2dgray_array)
    record.record(
        "_nothing_shapes",
        {},
        {
            "dummy_shapes": record.get_or_create_array_ref(layer1, shapes),
            "imarray": record.get_or_create_array_ref(layer2),
        },
        None,
    )
    record.save(Path(tmp_path / "test_shapes_replay"))
    record2: recorder.Recorder = recorder.Recorder()
    record2.load(Path(tmp_path / "test_shapes_replay"))
    result: Any = record2.replay(sample_2dgray_array)
    assert result is not None
    assert isinstance(result, int)
