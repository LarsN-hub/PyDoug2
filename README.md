# PyDoug Scientific Image Processing Software

## What is PyDoug?

PyDoug is a [napari](https://napari.org/stable/) plugin that adds a plethora of useful image processing and analysis widgets for n-dimensional scientific image processing. Most of the widgets included use functions from [numpy](https://numpy.org/) and [scikit-image](https://scikit-image.org/) with some additions from [algotom](https://myalgotomo.readthedocs.io/en/latest/index.html), [h5py](https://www.h5py.org/), [imageio](https://imageio.readthedocs.io/en/stable/), [matplotlib](https://matplotlib.org/), [porespy](https://porespy.org/index.html), and [scipy](https://scipy.org/). Combined, these widgets create a powerful and comprehensive scientific image processing and analysis software that is completely free and open source.

## Features

### No Programming Required

PyDoug will be able to launch from a single executable file (created with pyinstaller) for researchers who are not familiar with installing python packages (coming soon). For those with some coding experience or who already use napari and other image processing libraries, PyDoug is also easily operable from the command line with normal `pip` installation.

### Batch Processing

All operations performed on a single dataset are recorded and can be exported for automated batch processing of other image datasets later. Many *in situ* and *operando* imaging experiments collect multiple image datasets of the same sample in the same configuration while some process is occurring. With PyDoug, a suitable processing workflow can be developed for one dataset and then automatically applied to all other datasets.

### Add Your Own Functionality

Those with minimal programming experience can easily add their own functionality to PyDoug without having to understand the inner working of the napari GUI and widget assembly. In fact, most widgets that operate advanced image processing algorithms can be added to PyDoug with a single function. In this way, you can focus on the science and match while PyDoug automatically handles the widget implementation using the [magicui](https://pyapp-kit.github.io/magicgui/), [magic-class](https://hanjinliu.github.io/magic-class/), and [Qt for Python](https://wiki.qt.io/Qt_for_Python) libraries behind the scenes.

### Tooltips

All widgets and their included buttons and inputs have tooltips to explain what they do with links to the source library for a more detailed explanation. You can also automatically generate tooltips for your own added functions by simply including a numpy-style docstring at the top of your function.

### Built on Existing Libraries

PyDoug merely assembles functionality from other well-established libraries into an easy-to-use software. To learn how the GUI works, see the well-made documentation for [napari](https://napari.org/stable/usage.html). For more information on the functionality behind each widget, visit the links provided in each tooltip and read the documentation for each source yourself ([numpy](https://numpy.org/doc/stable/), [scikit-image](https://scikit-image.org/docs/stable/), [algotom](https://myalgotomo.readthedocs.io/en/latest/usage.html), [h5py](https://docs.h5py.org/en/stable/), [imageio](https://imageio.readthedocs.io/en/stable/), [matplotlib](https://matplotlib.org/stable/users/index), [porespy](https://porespy.org/_examples/index.html), [scipy](https://docs.scipy.org/doc/scipy/)).

## Tutorials

### Installing PyDoug

**Downloadable Executable Coming Soon**

**`pip` Install Coming Soon**

**Developers**: Clone the github repository to your computer and create a virtual python environment (`python -m venv PyDoug`). After activating the environment and navigating to the repository's directory, run the dev-setup.sh script (Windows users can do so using the git bash application) to install all required dependencies and to install PyDoug's packages in editable mode. Launch napari with `napari` in the terminal and activate the PyDoug plugin . If there are any problems, try reinstalling PyDoug's packages in editable mode (`pip install -e packages/pydoug-core --force-reinstall --no-deps` and `pip install -e packages/pydoug-napari[testing] --force-reinstall --no-deps`) as this is sometimes not done properly from the setup script.

### Adding Widgets

To add your own widgets, install PyDoug through the **Developers** route described above. Then, open the user_functions.py file found in the packages/pydoug-core/src/pydoug_core/user directory. In this file is a sample function (commented out) that will create a widget called "Simple Threshold" in a tab called "Users" in the PyDoug plugin. The following code is all that is required to create this widget:
```
from typing import Annotated

import numpy as np

from pydoug_core.registry import register_operation


@register_operation(
    category="user",
    display_name="Simple Threshold",
)
def sample_function(
    imarray: np.ndarray,
    Thresholds: tuple[
        Annotated[int, {"max": 65535, "min": 0}],
        Annotated[int, {"max": 65535, "min": 0}]
    ] = (50, 200),
) -> np.ndarray:
    """
    This is a numpy-style docstring for a sample function that thresholds an image array
    using inclusive upper and lower bounds and returns a binary image array where all
    pixels within the threshold range are white and all pixels outside the range are
    black.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be thresholded.
    Thresholds: tuple[int, int]
        The intensity values to use for the threshold range.

    Returns
    -------
    thresh_array: np.ndarray
        The thresholded image array.
    """

    thresh_array: np.ndarray = np.zeros(imarray.shape, np.uint8)
    thresh_array[(imarray <= max(Thresholds)) & (imarray >= min(Thresholds))] = 255
    return thresh_array
```

The widget appears in napari once PyDoug's plugin is launched like so (the tooltip popup over threshold is accessed by holding the mouse over the "Thresholds" label):

![The sample user widget](/assets/images/Sample_Widget.png)

The `@register_operation` decorator above the function is what tells PyDoug to add this function as a widget in napari. It specifies what tab to use, what the name of the widget should be, and can specify many other things such as how to handle masks, whether or not the operation should automatically be recorded in the workflow log, how to handle non-image array outputs (like plots and calculations), and more. Without this decorator, your function will not be added as a widget. Using this decorator automatically adds your function as a recordable operation that can be used in batch processing workflows without any additional coding.

In general, your function just needs to accept a numpy array (imarray) and return either a numpy array, numerical results (can be a single value or a dictionary of values, all are handled by PyDoug's results display widget), or a matplotlib plot (specifically must return a matplotlib.figure.Figure objet, do not use pyplot and do not show the figure in the function).

The type annotations for the input parameters are what tell magicgui what type of widget input should be created (e.g., the image array creates a layer drop-down selector while the tuple creates two inputs, and each is formatted using Annotated to only allow values in the range [0, 65535] and has default values of 50 and 200). There are many types of inputs that can be created, including toggleable checkboxes, single value floating point or integer inputs, string inputs, drop-down input selectors, slider bars, and more (see [magicgui](https://pyapp-kit.github.io/magicgui/type_map/) documentation for a comprehensive list).

The docstring at the top of the widget is what populates the tooltips automatically when the widget is created. Use the numpy style (short description followed by `Parameters` underlines and then `Returns` underlined) so that PyDoug can read it properly.

If there is some functionality that you want to include in the widget but are not sure how to specify this, simply look at the source code for other PyDoug widgets that use this functionality as examples. Use this sample widget as a template to get started.
