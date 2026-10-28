"""
Utilities for dynamically updating magicgui widget selections.
"""

from typing import Any

from magicgui.widgets import FunctionGui
from napari.viewer import Viewer


def connect_layer_autoselect(
    viewer: Viewer,
    widget: FunctionGui,
    param_name: str = "layer",
    autoselect: bool = True,
) -> None:
    """
    Keep magicgui layer selection widgets in sync with the napari viewer's layer list.

    Parameters
    ----------
    viewer: Viewer
        The napari viewer object.
    widget: FunctionGui
        The magicgui function.
    param_name: str
        The name of the parameter to be dynamically updated.
    autoselect: bool
        Whether or not to automatically update the layer selection.
    """

    from napari.utils.events import Event

    field: Any = getattr(widget, param_name)

    def _select_newest(event: Event):
        """
        Select the newest layer in the layers list while preventing incorrect layer
        types selection.

        Parameters
        ----------
        event: Event
            The napari gui event.
        """

        if event.value in field.choices:
            field.value = event.value

    def _select_last(event: Event):
        """
        Select the last layer in the layers list while preventing incorrect layer
        types selection.
        """

        if field.choices:
            field.value = field.choices[-1]

    viewer.layers.events.inserted.connect(field.reset_choices)
    viewer.layers.events.removed.connect(field.reset_choices)
    viewer.layers.events.reordered.connect(field.reset_choices)
    if autoselect:
        viewer.layers.events.inserted.connect(_select_newest)
        viewer.layers.events.removed.connect(_select_last)
        viewer.layers.events.reordered.connect(_select_last)
