"""
Batch processing widget for PyDoug's napari plugin.
"""

from pathlib import Path

from magicgui import magicgui
from qtpy.QtWidgets import QVBoxLayout, QWidget


def make_batch_widget(recorder) -> QWidget:
    """
    Create the batch widget container.

    Parameters
    ----------
    recorder: Recorder
        The parameters recording object.

    Returns
    -------
    container: QWidget
        The batch processing container widget.
    """

    @magicgui(call_button="Save Recording")
    def save_widget(filename: Path = Path("recording.json")) -> None:
        """
        Save the current parameters log.

        Parameters
        ----------
        filename: Path
            The file path where the parameters should be saved.
        """

        recorder.save(filename)

    @magicgui(call_button="Load Recording")
    def load_widget(filename: Path = Path("recording.json")) -> None:
        """
        Load parameters for batch processing.

        Parameters
        ----------
        filename: Path
            The path to the parameters file.
        """

        recorder.load(filename)

    @magicgui(call_button="Clear Recording")
    def clear_widget() -> None:
        """
        Clear the currently recorded steps.
        """

        recorder.steps.clear()

    container: QWidget = QWidget()
    layout: QVBoxLayout = QVBoxLayout()
    for w in (save_widget, load_widget, clear_widget):
        layout.addWidget(w.native)
    container.setLayout(layout)
    return container
