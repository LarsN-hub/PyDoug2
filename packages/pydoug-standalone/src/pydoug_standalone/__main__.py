"""
Main entry point for PyDoug's standalone executable app.
"""

import sys
from pathlib import Path


def _set_native_window_icon(hwnd: int, icon_path: Path) -> None:
    """
    Set the window's icon directly via the Win32 API for both small and large slots.

    Parameters
    ----------
    hwnd: int
        The window handle.
    icon_path: Path
        The path to the icon.
    """

    import ctypes

    IMAGE_ICON = 1
    LR_LOADFROMFILE = 0x10
    LR_DEFAULTSIZE = 0x40
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1

    hicon = ctypes.windll.user32.LoadImageW(
        None, str(icon_path), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE
    )
    if hicon:
        ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
        ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
    else:
        print("LoadImageW failed to load icon", file=sys.stderr)


def _get_icon_path() -> Path:
    """
    Resolve the path to the app icon.

    Returns
    -------
    icon_path: Path
        The path to the icon.
    """

    if getattr(sys, "frozen", False):
        base_dir: Path = Path(sys._MEIPASS)  # type: ignore
    else:
        base_dir: Path = Path(__file__).resolve().parents[3]
    return base_dir / "assets" / "images" / "logo.ico"


def main() -> None:
    """
    The main function. Launches a napari viewer and adds PyDoug's widgets to it.
    """

    if len(sys.argv) > 1 and sys.argv[1] == "--render-screenshot":
        from pydoug_napari._operations._screenshot_subprocess import main as render_main

        render_main(sys.argv[2])
        return

    """
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "PyDoug.ImageProcessing.Toolbox.1"
            )
        except Exception as exc:
            print(f"could not set AppUserModelID: {exc}", file=sys.stderr)
    """

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

    """
    try:
        hwnd = int(viewer.window._qt_window.winId())
        ico_path = _get_icon_path().with_suffix(".ico")
        if ico_path.exists():
            _set_native_window_icon(hwnd, ico_path)
        else:
            print(f"icon file not found: {ico_path}", file=sys.stderr)
    except Exception as exc:
        print(f"could not set native window icon: {exc}", file=sys.stderr)
    """

    viewer.window.add_dock_widget(
        MainWidget(viewer), name="PyDoug Image Processing Toolbox", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
