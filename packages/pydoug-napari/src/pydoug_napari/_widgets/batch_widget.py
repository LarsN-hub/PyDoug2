"""
Batch processing widget for PyDoug's napari plugin.
"""

from pathlib import Path

from magicclass.widgets import CollapsibleContainer, ScrollableContainer
from magicgui import magicgui

from pydoug_core import util


def make_batch_widget(recorder) -> ScrollableContainer:
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

    @magicgui(Save_Folder={"mode": "d"}, call_button="Save Workflow")
    def save_widget(
        Save_Folder: Path = Path(""), File_Name: str | Path = "workflow"
    ) -> None:
        """
        Save the current workflow.

        Parameters
        ----------
        Save_Folder: Path
            The folder where the workflow file should be saved.
        File_Name: str
            The name of the workflow file.
        """

        tidiedpath: Path = util.tidypath(Save_Folder / File_Name, "json")
        recorder.save(tidiedpath)

    @magicgui(call_button="Load Workflow")
    def load_widget(File_Path: Path = Path("")) -> None:
        """
        Load workflow for batch processing.

        Parameters
        ----------
        File_Path: Path
            The path to the parameters file.
        """

        recorder.load(File_Path)

    @magicgui(call_button="Clear Workflow")
    def clear_widget() -> None:
        """
        Clear the current workflow.
        """

        recorder.steps.clear()

    widget_list: list[CollapsibleContainer] = []
    for index, w in enumerate([save_widget, load_widget, clear_widget]):
        current_w: CollapsibleContainer = CollapsibleContainer(
            widgets=[w], labels=False
        )
        current_w.text = ["Save Workflow", "Load Workflow", "Clear Workflow"][index]
        widget_list.append(current_w)
    container: ScrollableContainer = ScrollableContainer(
        widgets=widget_list, labels=False
    )

    return container
