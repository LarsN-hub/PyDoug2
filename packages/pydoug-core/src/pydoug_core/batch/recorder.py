"""
Recorder for PyDoug operations used in napari widgets.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
from psygnal import Signal

from pydoug_core.io import imreader, imwriter
from pydoug_core.registry import OperationSpec, get_operation


@dataclass
class RecordedStep:
    """
    Define the operation name and its parameters for a recorded step.

    Attributes
    ----------
    step_id: str
        uuid reference to identify step.
    operation_name: str
        The name of the operation
    params: dict
        The dictionary of parameters and their values used in the operation.
    input_refs: dict[str, str]
        References to other steps as required for input parameters for an operation.
    layer_ref: None | Any
        The layer created during the recorded step.
    """

    step_id: str
    operation_name: str
    params: dict
    input_refs: dict[str, str] = field(default_factory=dict)
    layer_ref: None | Any = None


class Recorder:
    """
    Recorder object for recording operations and their parameters for batch processing.

    Attributes
    ----------
    steps: list[RecordedStep]
        List of recorded steps.
    _layer_to_step: WeakKeyDictionary
        Weak key dictionary mapping layers to steps.
    _external_layer_refs: WeakKeyDictionary
        Weak key dictionary mapping imported masks to layers.
    _arrays: dict[str, np.ndarray]
        Arrays that need be saved.
    """

    steps_changed: Signal = Signal()

    def __init__(self):
        self.steps: list[RecordedStep] = []
        self._layer_to_step: WeakKeyDictionary = WeakKeyDictionary()
        self._external_layer_refs: WeakKeyDictionary = WeakKeyDictionary()
        self._arrays: dict[str, np.ndarray] = {}

    def step_id_for_layer(self, layer) -> None | str:
        """
        Return the step_id that produced `layer`, or None if it's not a tracked step
        output (e.g., the originally loaded image).

        Parameters
        ----------
        layer
            The napari layer object.

        Returns
        -------
        step_id: None | str
            The uuid step_id reference or None if N/A.
        """

        if layer is not None:
            return self._layer_to_step.get(layer)
        return None

    def get_or_create_array_ref(self, layer: Any) -> str:
        """
        Reuse an existing reference for this layer if one was already assigned.

        Parameters
        ----------
        layer: the layer that needs to be referenced.

        Returns
        -------
        ref: str
            The reference to the layer.
        """

        existing: str | None = self._external_layer_refs.get(layer)
        if existing is not None:
            return existing
        tag: str = uuid.uuid4().hex[:8]
        ref: str = f"array:{tag}"
        self._external_layer_refs[layer] = ref
        self._arrays[tag] = np.copy(layer.data)
        return ref

    def register_array(self, array: np.ndarray) -> str:
        """
        Store an array directly and return its reference.

        Parameters
        ----------
        array: np.ndarray
            The array to be stored.

        Returns
        -------
        ref: str
            The reference for the array (with "array:" in front).
        """

        tag: str = uuid.uuid4().hex[:8]
        self._arrays[tag] = np.copy(array)
        return f"array:{tag}"

    def record(
        self,
        operation_name: str,
        params: dict,
        input_refs: dict[str, str],
        layer_ref: None | Any = None,
    ) -> str:
        """
        Record a step in the parameters log.

        Parameters
        ----------
        operation_name: str
            The name of the operation.
        params: dict
            The dictionary of the operation's parameters.
        input_refs: dict[str, str]
            The dictionary of references to other steps as required.
        layer_ref: None | Any
            The layer created during the recorded step.

        Returns
        -------
        step_id: str
            The step's string ID.
        """

        step_id: str = uuid.uuid4().hex
        self.steps.append(
            RecordedStep(
                step_id,
                operation_name,
                params,
                input_refs,
                layer_ref,
            )
        )
        if layer_ref is not None:
            self._layer_to_step[layer_ref] = step_id
        self.steps_changed.emit()
        return step_id

    def remove_step(self, step_id: str) -> None:
        """
        Remove any recorded step.

        Parameters
        ----------
        setp_id: str
            The step ID of the napari layer being removed.
        """

        new_steps: list[RecordedStep] = [
            s for s in self.steps if s.step_id is not step_id
        ]
        if len(new_steps) != len(self.steps):
            self.steps = new_steps
            stale: list[Any] = [
                layer for layer, sid in self._layer_to_step.items() if sid == step_id
            ]
            for layer in stale:
                del self._layer_to_step[layer]  # pragma: no cover
            self.steps_changed.emit()

    def remove_steps_by_layer(self, layer: Any) -> None:
        """
        Remove any recorded step whose output layer is `layer`.

        Parameters
        ----------
        layer: Any
            The layer of the step being removed.
        """

        new_steps: list[RecordedStep] = [
            s for s in self.steps if s.layer_ref is not layer
        ]
        if len(new_steps) != len(self.steps):
            self.steps = new_steps
            self._layer_to_step.pop(layer, None)
            self.steps_changed.emit()

    def dependents_of(self, step_id: str) -> list[RecordedStep]:
        """
        Return every step that depends on step_id.

        Parameters
        ----------
        setp_id: str
            The step's ID.

        Returns
        -------
        dependents_list: list[RecordedStep]
            The list of steps that depend on the step with ID step_id.
        """

        dependent_ids: set[str] = set()
        changed: bool = True
        while changed:
            changed = False
            frontier: set[str] = {step_id} | dependent_ids
            for step in self.steps:
                if step.step_id in frontier:
                    continue
                referenced: set[str] = {
                    ref.split(":", 1)[1]
                    for ref in step.input_refs.values()
                    if ref.startswith("step:")
                }
                if referenced & frontier:
                    dependent_ids.add(step.step_id)
                    changed = True
        return [s for s in self.steps if s.step_id in dependent_ids]

    def clear(self) -> None:
        """
        Clear the recorded steps.
        """

        self.steps = []
        self._layer_to_step.clear()
        self._external_layer_refs.clear()
        self._arrays.clear()
        self.steps_changed.emit()

    def save(self, path: Path) -> None:
        """
        Save a parameters log as a json file and arrays folder.

        Parameters
        ----------
        path: Path
            The path to save the workflow to.
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays_dir: Path = path / "arrays"
        used_tags: set[str] = {
            ref.split(":", 1)[1]
            for step in self.steps
            for ref in step.input_refs.values()
            if ref.startswith("array:")
        }

        data: list[dict] = []
        for s in self.steps:
            refs: dict[str, str] = {
                name: (
                    f"file:arrays/{ref.split(':', 1)[1]}.tiff"
                    if ref.startswith("array:")
                    else ref
                )
                for name, ref in s.input_refs.items()
            }
            data.append(
                {
                    "step_id": s.step_id,
                    "operation": s.operation_name,
                    "params": s.params,
                    "input_refs": refs,
                }
            )

        for tag in used_tags:
            arrays_dir.mkdir(parents=True, exist_ok=True)
            imwriter.imwrite(self._arrays[tag], arrays_dir, tag, "tiff")
            written = arrays_dir / f"{tag}.tiff"
            if not written.exists():
                raise RuntimeError(f"Failed to save array for tag '{tag}' to {written}")

        (path / "workflow.json").write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """
        Load a workflow from its directory.

        Parameters
        ----------
        path: Path
            The path to the workflow directory to be loaded.
        """

        path = Path(path)
        data: Any = json.loads((path / "workflow.json").read_text())
        self._layer_to_step.clear()
        self._external_layer_refs.clear()
        self._arrays.clear()
        tag_by_file: dict[str, str] = {}

        def _convert_ref(ref: str) -> str:
            if not ref.startswith("file:"):
                return ref
            rel: str = ref.split(":", 1)[1]
            if rel not in tag_by_file:
                array: np.ndarray | None = imreader.imread(path / rel)
                if array is None:
                    raise RuntimeError(
                        f"Could not read referenced array file: {path / rel}"
                    )
                tag_by_file[rel] = self.register_array(array)
            return tag_by_file[rel]

        self.steps: list[RecordedStep] = [
            RecordedStep(
                d["step_id"],
                d["operation"],
                d["params"],
                {
                    name: _convert_ref(ref)
                    for name, ref in d.get("input_refs", {}).items()
                },
            )
            for d in data
        ]

        self.steps_changed.emit()

    def _resolve(
        self,
        ref: str,
        primary: np.ndarray,
        outputs: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """
        Resolve input reference array.

        Parameters
        ----------
        ref: str
            The array's reference string.
        primary: np.ndarray
            The primary array.
        outputs: dict[str, object]
            Other output arrays.

        Returns
        -------
        resolution: np.ndarray | None
            The resolved array.
        """

        if ref == "primary":
            return primary
        if ref.startswith("step:"):
            return outputs[ref.split(":", 1)[1]]
        if ref.startswith("array:"):
            return self._arrays[ref.split(":", 1)[1]]
        raise ValueError(f"Unrecognized input reference: {ref!r}")

    def iter_replay(self, imarray: np.ndarray):
        """
        Yield (step, spec, result) for each recorded step in order.

        Parameters
        ----------
        imarray: np.ndarray
            The image array being processed.

        Yields
        ------
        step: RecordedStep
            The current recorded step.
        spec: OperationSpec
            The operation specifications for the current step.
        result: Any
            The result of the operaiton (array, alphanumeric, or plot figure)
        """

        outputs: dict[str, np.ndarray] = {}
        for step in self.steps:
            spec: OperationSpec = get_operation(step.operation_name)
            other_kwargs = {
                name: self._resolve(ref, imarray, outputs)
                for name, ref in step.input_refs.items()
                if name != "layer"
            }
            if "layer" in step.input_refs:
                primary_arg = self._resolve(step.input_refs["layer"], imarray, outputs)
                result = spec.func(primary_arg, **other_kwargs, **step.params)
            else:
                result = spec.func(**other_kwargs, **step.params)
            outputs[step.step_id] = result
            yield step, spec, result

    def replay(self, imarray: np.ndarray) -> np.ndarray | dict | Any:
        """
        Apply a recorded workflow to an image.

        Parameters
        ----------
        imarray: np.ndarray
            The image array to apply the workflow to.
        """

        result: Any = None
        for _, _, result in self.iter_replay(imarray):
            pass
        return result
