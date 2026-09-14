"""
Utilities for dynamically updating magicgui widget selections.
"""


def connect_layer_autoselect(
    viewer, widget, param_name: str = "layer", autoselect: bool = True
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

    field = getattr(widget, param_name)

    def _select_newest(event):
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

    def _select_last():
        """
        Select the last layer in the layers list.

        Parameters
        ----------
        event: Event
            The napari gui event.
        """

        if viewer.layers:
            field.value = viewer.layers[-1]

    viewer.layers.events.inserted.connect(field.reset_choices)
    viewer.layers.events.removed.connect(field.reset_choices)
    viewer.layers.events.reordered.connect(field.reset_choices)
    if autoselect:
        viewer.layers.events.inserted.connect(_select_newest)
        viewer.layers.events.removed.connect(_select_last)
        viewer.layers.events.reordered.connect(_select_last)
