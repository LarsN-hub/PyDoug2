"""
Operations recording log display widget for PyDoug's napari plugin.
"""

from qtpy.QtCore import QSize
from qtpy.QtWidgets import QListWidget

from pydoug_core.batch.recorder import Recorder


class RecordingLogWidget(QListWidget):
    """
    A constant display of the parameters log as a separate widget in napari.

    Attributes
    ----------
    _recorder: Recorder
        The workflow recorder object.
    """

    def __init__(self, recorder: Recorder) -> None:

        super().__init__()
        recorder.steps_changed.connect(self._refresh)
        self._recorder: Recorder = recorder
        self._refresh()

    def _refresh(self) -> None:
        """
        Refresh the workflow log.
        """

        self.clear()
        for i, step in enumerate(self._recorder.steps):
            self.addItem(f"{i}: {step.operation_name}({step.params})")

    def sizeHint(self):
        return QSize(200, self.minimumHeight())
