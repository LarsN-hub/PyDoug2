## 0.1.3

### Updates
* Expanded CI tests matrix to cover python 3.12 and lower dependency versions.
* Updated third party licenses to cover 0.1.1 additions.

## 0.1.2

### Additions
* Created and populated CHANGELOG.md.
* Created GitHub workflow (publish.yml) to automatically publish GitHub releases to PyPI.
* Created version checking script (scripts/check_versions.py) to validate publish workflow.

## 0.1.1

### Additions
* Calculate fractal dimension with porespy's boxcount function ("Measure" tab).
* Import additional image formats with astropy, imagecodecs, and simpleitk packages.
* Improve napari Qt performance with PyOpenGL-accelerate package.

### Updates
* Fixed bug where scikit-image's graph module (path metrics widget in "Features" tab) was not included in the standalone version by adding scikit-image's metadata to pyinstaller build.
* Fixed bug with certain tiff compression formats not importing by adding the imagecodecs package.

## 0.1.0

### First release!
