"""
Utilities for dynamically updating magicgui widget selections.
"""


def connect_layer_autoselect(viewer, widget, param_name: str = "layer") -> None:
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
    """

    field = getattr(widget, param_name)

    def _select_newest(event):
        """
        Guard against selecting incorrect layer types.

        Parameters
        ----------
        event: Event
            The napari gui event.
        """

        if event.value in field.choices:
            field.value = event.value

    viewer.layers.events.inserted.connect(field.reset_choices)
    viewer.layers.events.removed.connect(field.reset_choices)
    viewer.layers.events.reordered.connect(field.reset_choices)
    viewer.layers.events.inserted.connect(_select_newest)
