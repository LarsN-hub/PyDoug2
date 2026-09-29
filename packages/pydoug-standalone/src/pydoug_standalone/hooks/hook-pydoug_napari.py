"""
Hook for PyDoug's napari plugin package.
"""

from PyInstaller.utils.hooks import collect_all, copy_metadata


def hook(hook_api):
    datas, binaries, hiddenimports = collect_all("pydoug_napari")
    datas += copy_metadata("pydoug_napari")
    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)
