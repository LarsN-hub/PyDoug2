"""
Hook for PyDoug's core package.
"""

from PyInstaller.utils.hooks import collect_all, copy_metadata


def hook(hook_api):
    datas, binaries, hiddenimports = collect_all("pydoug_core")
    datas += copy_metadata("pydoug_core")
    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)
