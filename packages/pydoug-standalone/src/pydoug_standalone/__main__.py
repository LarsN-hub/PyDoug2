"""
Main entry point for PyDoug standalone executable app
"""

import sys


def main() -> None:
    """
    The main function. Launches a napari viewer and adds PyDoug's widgets to it.
    """

    if len(sys.argv) > 1 and sys.argv[1] == "--render-screenshot":
        from pydoug_napari._operations._screenshot_subprocess import main as render_main

        render_main(sys.argv[2])
        return

    import napari

    from pydoug_napari._widgets.main_widget import MainWidget

    viewer: napari.Viewer = napari.Viewer()
    viewer.window.add_dock_widget(
        MainWidget(viewer), name="PyDoug Image Processing Toolbox", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
