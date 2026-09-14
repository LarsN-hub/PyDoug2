"""
Recorder for PyDoug operations used in napari widgets.
"""

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from numpy import ndarray
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
    pending_masks: dict
        Masks used in operations that may need to be saved for batch processing.
    """

    step_id: str
    operation_name: str
    params: dict
    input_refs: dict[str, str] = field(default_factory=dict)
    layer_ref: None | Any = None
    pending_masks: dict[str, ndarray] = field(default_factory=dict)


class Recorder:
    """
    Recorder object for recording operations and their parameters for batch processing.

    Attributes
    ----------
    steps: list[RecordedStep]
        List of recorded steps.
    _layer_to_step: WeakKeyDictionary
    """

    steps_changed: Signal = Signal()

    def __init__(self):
        self.steps: list[RecordedStep] = []
        self._layer_to_step: WeakKeyDictionary = WeakKeyDictionary()
        self._workflow_dir: Path | None = None

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

        return self._layer_to_step.get(layer)

    def record(
        self,
        operation_name: str,
        params: dict,
        input_refs: dict[str, str],
        layer_ref: None | Any = None,
        pending_masks: dict | None = None,
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
        pending_masks: dict | None
            Pending masks to be saved if required.
        """

        step_id: str = uuid.uuid4().hex
        self.steps.append(
            RecordedStep(
                step_id,
                operation_name,
                params,
                input_refs,
                layer_ref,
                pending_masks or {},
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
                del self._layer_to_step[layer]
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
        self.steps_changed.emit()

    def save(self, path: Path) -> None:
        """
        Save a parameters log as a json file and masks folder.

        Parameters
        ----------
        path: Path
            The path to save the workflow to.
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        masks_dir: Path = path / "masks"
        data: list[dict] = []

        for s in self.steps:
            refs: dict = dict(s.input_refs)

            for name, ref in list(refs.items()):
                if ref.startswith("file:") and self._workflow_dir is not None:
                    src: Path = self._workflow_dir / ref.split(":", 1)[1]
                    fname: str = Path(ref.split(":", 1)[1]).name
                    masks_dir.mkdir(exist_ok=True)
                    dest: Path = masks_dir / fname
                    if src.resolve() != dest.resolve():
                        shutil.copy2(src, dest)
                    refs[name] = f"file:masks/{fname}"

            for name, arr in s.pending_masks.items():
                masks_dir.mkdir(exist_ok=True)
                fname = f"{s.step_id}_{name}.tiff"
                imwriter.imwrite(arr, masks_dir, f"{s.step_id}_{name}", "tiff")
                refs[name] = f"file:masks/{fname}"

            s.input_refs = refs
            s.pending_masks = {}
            data.append(
                {
                    "step_id": s.step_id,
                    "operation": s.operation_name,
                    "params": s.params,
                    "input_refs": refs,
                }
            )

        (path / "workflow.json").write_text(json.dumps(data, indent=2))
        self._workflow_dir = path

    def load(self, path: Path) -> None:
        """
        Load a parameters log from a json file.

        Parameters
        ----------
        path: Path
            The path to the parameters log to be loaded.
        """

        path = Path(path)
        data: Any = json.loads((path / "workflow.json").read_text())
        self.steps: list[RecordedStep] = [
            RecordedStep(
                d["step_id"], d["operation"], d["params"], d.get("input_refs", {})
            )
            for d in data
        ]
        self._layer_to_step.clear()
        self._workflow_dir = path
        self.steps_changed.emit()

    def _resolve(
        self,
        ref: str,
        step: RecordedStep,
        param_name: str,
        primary: ndarray,
        outputs: dict[str, ndarray],
    ) -> ndarray | None:
        """
        Resolve input reference array.

        Parameters
        ----------
        ref: str
            The array's reference string.
        step: RecordedStep
            The array's recorded step.
        param_name: str
            The name of the array's parameter.
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
        if ref.startswith("pending:"):
            return step.pending_masks[param_name]
        if ref.startswith("file:"):
            if self._workflow_dir is None:
                raise ValueError(
                    f"Cannot resolve '{ref}':"
                    "no workflow directory is set (was load() called?)"
                )
            return imreader.imread(self._workflow_dir / ref.split(":", 1)[1])
        raise ValueError(f"Unrecognized input reference: {ref!r}")

    def replay(self, imarray: ndarray) -> ndarray | dict | Any:
        """
        Apply a recorded workflow to an image.

        Parameters
        ----------
        imarray: np.ndarray
            The image array to apply parameters to.
        """

        outputs: dict[str, ndarray] = {}
        result: Any = None
        for step in self.steps:
            spec: OperationSpec = get_operation(step.operation_name)
            primary_ref: str = step.input_refs.get("layer", "primary")
            primary_arg: ndarray | None = self._resolve(
                primary_ref, step, "layer", imarray, outputs
            )
            other_kwargs = {
                name: self._resolve(ref, step, name, imarray, outputs)
                for name, ref in step.input_refs.items()
                if name != "layer"
            }
            result = spec.func(primary_arg, **other_kwargs, **step.params)
            outputs[step.step_id] = result
        return result
