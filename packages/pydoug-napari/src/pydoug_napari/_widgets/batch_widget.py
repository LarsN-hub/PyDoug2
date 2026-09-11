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

    @magicgui(Parent_Folder={"mode": "d"}, call_button="Save Workflow")
    def save_widget(
        Parent_Folder: Path = Path(""), Workflow_Folder_Name: str = "Workflow"
    ) -> None:
        """
        Save the current workflow.

        Parameters
        ----------
        Save_Folder: Path
            The folder where the workflow folder should be saved.
        File_Name: str
            The name of the workflow folder.
        """

        tidiedpath: Path = util.tidypath(Parent_Folder / Workflow_Folder_Name, "dir")
        recorder.save(tidiedpath)

    @magicgui(Workflow_Folder_Path={"mode": "d"}, call_button="Load Workflow")
    def load_widget(Workflow_Folder_Path: Path = Path("")) -> None:
        """
        Load workflow for batch processing.

        Parameters
        ----------
        File_Path: Path
            The path to the workflow folder.
        """

        recorder.load(Workflow_Folder_Path)

    @magicgui(call_button="Clear Workflow")
    def clear_widget() -> None:
        """
        Clear the current workflow.
        """

        recorder.clear()

    def _get_collapsible_widget(widget) -> CollapsibleContainer:
        return CollapsibleContainer(widgets=[widget], labels=False)

    widget_list: list[CollapsibleContainer] = []
    for index, w in enumerate([save_widget, load_widget, clear_widget]):
        current_w: CollapsibleContainer = _get_collapsible_widget(w)
        current_w.text = ["Save Workflow", "Load Workflow", "Clear Workflow"][index]
        current_w.native.setObjectName("Collapsible")
        current_w.native.setStyleSheet(
            "QWidget#Collapsible { border: 1px solid #888; border-radius: 4px; }"
        )
        current_w.native.setContentsMargins(2, 2, 2, 6)
        widget_list.append(current_w)
    container: ScrollableContainer = ScrollableContainer(
        widgets=widget_list, labels=False
    )

    return container
