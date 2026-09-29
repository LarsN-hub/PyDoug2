"""
Hook for porespy package.
"""

from PyInstaller.utils.hooks import collect_all, copy_metadata


def hook(hook_api):
    datas, binaries, hiddenimports = collect_all("porespy")
    datas += copy_metadata("porespy")
    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)
