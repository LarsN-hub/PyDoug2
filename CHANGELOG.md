## 0.1.4

### Updates
* Updated README to reflect recent changes.
* Fixed size distribution widget counting background as a label.
* Fixed missing imports in standalone build.spec file.
* Fixed bug in axial and value gradient color widgets and removed image layer output option.

## 0.1.3

### Additions
* Added image to labels and labels to image widgets in "Visualize" tab.

### Updates
* Fixed bug where boolean input images would crash watershed labeling if "Use Negative Image" was selected.
* Expanded CI tests matrix to cover python 3.12 and lower dependency versions.
* Updated third party licenses to cover 0.1.1 additions.
* Increased maximum input range for size/distance of remove objects and remove holes widgets.
* Added image vs. labels return type selection for remove objects widget.

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
