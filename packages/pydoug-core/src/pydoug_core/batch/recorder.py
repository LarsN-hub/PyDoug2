"""
Recorder for PyDoug operations used in napari widgets.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from numpy import ndarray

from pydoug_core.registry import get_operation


@dataclass
class RecordedStep:
    """
    Define the operation name and its parameters for a recorded step.

    Attributes
    ----------
    operation_name: str
        The name of the operation
    params: dict
        The dictionary of parameters and their values used in the operation.
    """

    operation_name: str
    params: dict


class Recorder:
    """
    Recorder object for recording operations and their parameters for batch processing.

    Attributes
    ----------
    steps: list[RecordedStep]
        List of recorded steps.
    """

    def __init__(self):
        self.steps: list[RecordedStep] = []

    def record(self, operation_name: str, params: dict) -> None:
        """
        Record a step in the parameters log.

        Parameters
        ----------
        operation_name: str
            The name of the operation.
        params: dict
            The dictionary of the operation's parameters.
        """

        self.steps.append(RecordedStep(operation_name, params))

    def save(self, path: Path) -> None:
        """
        Save a parameters log as a json file.

        Parameters
        ----------
        path: Path
            The path to save the parameters log to.
        """

        data = [{"operation": s.operation_name, "params": s.params} for s in self.steps]
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
        self.steps = [RecordedStep(d["operation"], d["params"]) for d in data]

    def replay(self, imarray: ndarray) -> ndarray:
        """
        Apply a parameters log to an image.

        Parameters
        ----------
        imarray: np.ndarray
            The image array to apply parameters to.
        """

        result = imarray
        for step in self.steps:
            result = get_operation(step.operation_name).func(result, **step.params)
        return result
