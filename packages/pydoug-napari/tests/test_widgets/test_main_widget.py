"""
Tests for PyDoug's napari plugin main widget module.
"""

from napari.viewer import Viewer

from pydoug_napari._widgets import main_widget


def test_make_main_widget(make_napari_viewer) -> None:

    viewer: Viewer = make_napari_viewer()
    widget: main_widget.MainWidget = main_widget.MainWidget(viewer)
    assert widget.count() >= 1
