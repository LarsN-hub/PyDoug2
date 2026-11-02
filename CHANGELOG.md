## Unreleased

### Updates
*

### Additions
*

## 0.1.1

### Updates
* Fixed bug where scikit-image's graph module (path metrics widget in "Features" tab) was not included in the standalone version by adding scikit-image's metadata to pyinstaller build.
* Fixed bug with certain tiff compression formats not importing by adding the imagecodecs package.

### Additions
* Calculate fractal dimension with porespy's boxcount function ("Measure" tab).
* Import additional image formats with astropy, imagecodecs, and simpleitk packages.
* Improve napari Qt performance with PyOpenGL-accelerate package.

## 0.1.0

### First release!
