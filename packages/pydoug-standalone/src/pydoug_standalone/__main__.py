"""
Main entry point for PyDoug's standalone executable app.
"""

import sys
from pathlib import Path


def _get_icon_path() -> Path:
    """
    Resolve the path to the app icon.
    """

    if getattr(sys, "frozen", False):
        base_dir: Path = Path(sys._MEIPASS)  # type: ignore
    else:
        base_dir: Path = Path(__file__).resolve().parents[3]
    return base_dir / "assets" / "images" / "logo.png"


def main() -> None:
    """
    The main function. Launches a napari viewer and adds PyDoug's widgets to it.
    """

    if len(sys.argv) > 1 and sys.argv[1] == "--render-screenshot":
        from pydoug_napari._operations._screenshot_subprocess import main as render_main

        render_main(sys.argv[2])
        return

    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "PyDoug.ImageProcessing.Toolbox.1"
            )
        except Exception as exc:
            print(f"could not set AppUserModelID: {exc}", file=sys.stderr)

    import napari
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    from pydoug_napari._widgets.main_widget import MainWidget

    viewer: napari.Viewer = napari.Viewer()

    try:
        app = QApplication.instance()
        if app is not None:
            icon_path = _get_icon_path()
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))  # type: ignore
    except Exception as exc:
        print(f"could not set window icon: {exc}", file=sys.stderr)

    viewer.window.add_dock_widget(
        MainWidget(viewer), name="PyDoug Image Processing Toolbox", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
