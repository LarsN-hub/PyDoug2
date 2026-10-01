# -*- mode: python ; coding: utf-8 -*-

from importlib.util import find_spec
from pathlib import Path

assert Path("packages/pydoug-core/src/pydoug_core/_manifest.py").exists()
assert Path("packages/pydoug-napari/src/pydoug_napari/_operations/_manifest.py").exists()

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

datas, binaries, hiddenimports = [], [], []

for pkg in (
    "app_model",
    "cmasher",
    "imageio",
    "magicgui",
    "magicclass",
    "napari",
    "napari_builtins",
    "napari_console",
    "napari_metadata",
    "napari_resources",
    "napari_svg",
    "npe2",
    "qtpy",
    "superqt",
    "urllib3",
    "vispy",
):
    if find_spec(pkg) is not None:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h

for pkg in ("pydoug_core", "pydoug_napari", "pydoug_standalone"):
    hiddenimports += collect_submodules(pkg)
    datas += collect_data_files(pkg)
datas += copy_metadata("pydoug-core") + copy_metadata("pydoug-napari")
datas += [
    ('assets/images/logo.png', 'assets/images'),
    ('assets/images/logo.ico', 'assets/images'),
]

a = Analysis(
    ['packages/pydoug-standalone/src/pydoug_standalone/__main__.py'],
    pathex=[
        'packages/pydoug-core/src',
        'packages/pydoug-napari/src',
        'packages/pydoug-standalone/src',
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath = ["./packages/pydoug-standalone/src/pydoug_standalone/hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PyDoug',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon = "./assets/images/logo.ico"
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PyDoug',
)
