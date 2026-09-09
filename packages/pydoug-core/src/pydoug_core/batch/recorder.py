"""
Recorder for PyDoug operations used in napari widgets.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from numpy import ndarray
from psygnal import Signal

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
    """

    steps_changed: Signal = Signal()

    def __init__(self):
        self.steps: list[RecordedStep] = []
        self._layer_to_step: WeakKeyDictionary = WeakKeyDictionary()

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
        """

        step_id: str = uuid.uuid4().hex
        self.steps.append(
            RecordedStep(step_id, operation_name, params, input_refs, layer_ref)
        )
        if layer_ref is not None:
            self._layer_to_step[layer_ref] = step_id
        self.steps_changed.emit()
        return step_id

    def remove_steps_by_layer(self, layer: object) -> None:
        """
        Remove any recorded step whose output layer is "layer".

        Parameters
        ----------
        layer: object
            The napari layer being removed.
        """

        new_steps: list[RecordedStep] = [
            s for s in self.steps if s.layer_ref is not layer
        ]
        if len(new_steps) != len(self.steps):
            self.steps = new_steps
            self._layer_to_step.pop(layer, None)
            self.steps_changed.emit()

    def clear(self) -> None:
        """
        Clear the recorded steps.
        """

        self.steps = []
        self._layer_to_step.clear()
        self.steps_changed.emit()

    def save(self, path: Path) -> None:
        """
        Save a parameters log as a json file.

        Parameters
        ----------
        path: Path
            The path to save the parameters log to.
        """

        data: list[dict] = [
            {
                "step_id": s.step_id,
                "operation": s.operation_name,
                "params": s.params,
                "input_refs": s.input_refs,
            }
            for s in self.steps
        ]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """
        Load a parameters log from a json file.

        Parameters
        ----------
        path: Path
            The path to the parameters log to be loaded.
        """

        data = json.loads(Path(path).read_text())
        self.steps: list[RecordedStep] = [
            RecordedStep(
                d["step_id"], d["operation"], d["params"], d.get("input_refs", {})
            )
            for d in data
        ]
        self._layer_to_step.clear()
        self.steps_changed.emit()

    def replay(self, imarray: ndarray):
        """
        Apply a recorded workflow to an image.

        Parameters
        ----------
        imarray: np.ndarray
            The image array to apply parameters to.
        """

        outputs: dict[str, object] = {}
        result: Any = None
        for step in self.steps:
            spec: OperationSpec = get_operation(step.operation_name)
            ref: str = step.input_refs.get("layer", "primary")
            arg: ndarray | object = (
                imarray if ref == "primary" else outputs[ref.split(":", 1)[1]]
            )
            result = spec.func(arg, **step.params)
            outputs[step.step_id] = result
        return result
