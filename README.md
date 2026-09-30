# PyDoug Scientific Image Processing Software

## What is PyDoug?

PyDoug is a [napari](https://napari.org/stable/) plugin that adds a plethora of useful image processing and analysis widgets for n-dimensional scientific image processing. Most of the widgets included use functions from [numpy](https://numpy.org/) and [scikit-image](https://scikit-image.org/) with some additions from [algotom](https://myalgotomo.readthedocs.io/en/latest/index.html), [h5py](https://www.h5py.org/), [imageio](https://imageio.readthedocs.io/en/stable/), [matplotlib](https://matplotlib.org/), [porespy](https://porespy.org/index.html), and [scipy](https://scipy.org/). Combined, these widgets create a powerful and comprehensive scientific image processing and analysis software that is completely free and open source. Additionally, it can be operated from a single downloadable file without any coding or installation required for scientists without programming experience.

I made PyDoug because I was inspired by the [MATBOX Microstructure Analysis Toolbox](https://github.com/NatLabRockies/MATBOX_Microstructure_analysis_toolbox) but wanted a python-based free and open-source image processing toolbox in case I ever lost access to MATLAB. While MATBOX is more powerful for experienced image processing researchers, I hope that PyDoug will serve as a useful for tool for scientists who prefer python (or who do not have access to MATLAB) and as a beginner-friendly option for those who are newer to image processing or who have no programming experience. Additionally, PyDoug is built in the [napari GUI](https://napari.org/stable/) which adds some visualization capabilities that are more difficult to achieve in MATLAB-based programs.

## Features

### No Programming Required

PyDoug can launch from a single executable file (available in the releases section, created with [pyinstaller](https://pyinstaller.org/en/stable/)) for researchers who are not familiar with installing python packages. For those with some coding experience or who already use napari and other image processing libraries, PyDoug is also easily operable from the command line with normal `pip` installation and can be added to napari as a plugin.

### Batch Processing

All operations performed on a single dataset are recorded and can be exported for automated batch processing of other image datasets later. Many *in situ* and *operando* imaging experiments collect multiple image datasets of the same sample in the same configuration while some process is occurring; with PyDoug, a suitable processing workflow can be developed for one dataset and then automatically applied to all other datasets.

### Add Your Own Functionality

Those with minimal programming experience can easily add their own functionality to PyDoug without having to understand the inner working of the napari GUI and widget assembly. In fact, most widgets that operate advanced image processing algorithms can be added to PyDoug with a single function. In this way, you can focus on the science and math while PyDoug automatically handles the widget implementation using the [magicui](https://pyapp-kit.github.io/magicgui/), [magic-class](https://hanjinliu.github.io/magic-class/), [napari](https://napari.org/stable/), and [Qt for Python](https://wiki.qt.io/Qt_for_Python) libraries behind the scenes.

### Tooltips

All widgets and their included buttons and inputs have tooltips to explain what they do with links to the source library for a more detailed explanation. You can also automatically generate tooltips for your own added functions by simply including a [numpy-style docstring](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html) at the top of your function.

### Built on Existing Libraries

PyDoug assembles functionality from other well-established libraries into an easy-to-use software. To learn how the GUI works, see the well-made documentation for [napari](https://napari.org/stable/usage.html). For more information on the functionality behind each widget, visit the links provided in each tooltip and read the documentation for each source yourself ([numpy](https://numpy.org/doc/stable/), [scikit-image](https://scikit-image.org/docs/stable/), [algotom](https://myalgotomo.readthedocs.io/en/latest/usage.html), [h5py](https://docs.h5py.org/en/stable/), [imageio](https://imageio.readthedocs.io/en/stable/), [matplotlib](https://matplotlib.org/stable/users/index), [porespy](https://porespy.org/_examples/index.html), [scipy](https://docs.scipy.org/doc/scipy/)).

## Tutorials

### Installing PyDoug

**Single Executable (No Coding or Install Required):**

Note, currently only available on Windows, Linux coming soon!
1. Navigate to the "Releases" section of the GitHub page.
2. Find the most recent release.
3. Download the "PyDoug.zip" file from the "Assets" section at the bottom of the release.
4. Move the zip file to wherever you would like and extract it.
5. After extraction there will be a new folder named "PyDoug" with the "PyDoug.exe" executable file inside next to a folder named "_internal". Run PyDoug by double-clicking this executable file (no program install is required, everything runs out of the "_internal" folder).
6. IMPORTANT: DO NOT remove the .exe file or the "_internal" folder from the PyDoug folder.
7. The .exe file can be pinned to your Start menu or taskbar for easy access if you wish.

**`pip` Install Coming Soon**

**Developers**:

Clone the github repository to your computer and create a virtual python environment (`python -m venv PyDoug_Dev` on Windows). After activating the environment and navigating to the repository's directory, run the dev-setup.sh script (Windows users can do so using the git bash application) to install all required dependencies and to install PyDoug's packages in editable mode. Launch napari with `napari` in the terminal in the activated python environment and then activate the PyDoug plugin from napari's "Plugins" menu on the top left. If there are any problems, try re-installing PyDoug's packages in editable mode (`pip install -e packages/pydoug-core --force-reinstall --no-deps` and `pip install -e packages/pydoug-napari[testing] --force-reinstall --no-deps`) as this is sometimes not accomplished properly from the setup script.

### Using PyDoug

This section will cover some basic napari usage (more can be read on [napari's website]((https://napari.org/stable/usage.html))) as well as basic PyDoug widget usage.

When PyDoug is launched (either through the executable file or by selecting the PyDoug plugin if launching from napari as a developer) the following screen will appear. Several tabs containing PyDoug's widgets appear on the right with two logs on the bottom, one for the workflow (records every step you perform) and one for any numerical results (displays any calculations you perform).

![Startup Screen](/assets/images/Startup_Screen.png)

If you add an image to the napari canvas (this can be done through the file menu in the top left or just by dragging and dropping an image file onto the canvas) it will be rendered (in 2D by default) in the canvas as seen below. This will add the image to the layer list on the bottom left. Clicking on the layer in the layer list will bring up controls for this layer in the controls tab above the layer list (here things like the contrast limits can be changed for ease of viewing). The image can be removed by clicking the garbage can icon above the layer list with the layer selected. The visibility of the layer can be toggled with the eyeball button on the layer in the layer list.

![Loaded Image Layer](/assets/images/Loaded_Image.png)

The image we have loaded here for examples is a 3D image and can be viewed in 3D by clicking the 2D/3D toggle button on the bottom left below the layer list as seen below. Whether in 2D or 3D, the image layer can be moved around the screen by clicking and dragging and can be zoomed in on with the mouse wheel. Holding down shift before dragging in 3D will move the layer in 3D space instead of rotating it about the center as occurs when drag clicking without the shift key.

![3D Image Layer](/assets/images/3D_Layer.png)

Let's perform an operation on the image layer. Below is a zoomed in image of one of the widgets from the "Transformations" tab in the widgets panel on the right, the "Rotate" widget. This allows us to rotate the image about its center. Almost all widgets will have a "Layer" drop-down selection on top to select which layer the operation is being performed on. Then follows the options for the operation. For the rotate widget, we have the angle to rotate the image by, a checkbox signaling whether or not to rotate clockwise or counter-clockwise, a checkbox to toggle resizing the image to fully fit the new rotated image, and a checkbox to preserve the original [data type](https://scikit-image.org/docs/stable/user_guide/data_types.html) of the image (integer values are converted to floating point values for some operations, but can be converted back using this option). We will rotate the image by 45 degrees counter-clockwise without resizing but preserving the original data type.

![Rotate Widget](/assets/images/Rotate_Widget.png)

Below is the new image layer created after our rotation operation. This operation created a new image layer that was added to the top of the layer list on the left and recorded the parameters used in the operation in the workflow log on the bottom. If we decide we don't like the operation we performed, we can delete the image layer and its log in the workflow will be automatically removed as well.

![Rotated Image](/assets/images/Rotated_Image_Layer.png)

The original image layer is still there, just hidden behind the new top-most image layer. We can view it again by toggling the visibility of the new image layer as shown below. Because it all image layers remain present, workflows can be dynamically designed and tested by performing operations and then removing their corresponding image layer (acting as an "undo") before settling on a final workflow for batch processing.

![Toggled Image Layer](/assets/images/Toggled_Layer.png)

Not all operations produce a new image layer. Some, like the statistics calculation widget in the "Measurements" tab produce a numerical result (in this case, several statistics about the grayscale values of the image layer). When these operations are performed, their results are recorded in the results log in the bottom right like so:

![Stats Calculation](/assets/images/Stats_Calculation.png)

The workflow log and the results log begin quite small by default so as to stay out of the way of the canvas, but they can be expanded by dragging the top area of them upwards or even undocked from their original location for display on another monitor. In fact, all widgets on the canvas can be undocked and moved around or re-docked in a new location by simply clicking them and dragging them away as shown below. Now we can see the full list of results gathered from our statistics calculation.

![Undocked Widgets](/assets/images/Undocked_Widgets.png)

Once you have processed an image and gotten it to a final state that you like, you can save your workflow and apply it automatically to any other images you wish to process in the "Batch" tab. The top widget in the batch tab is for saving a workflow to a file on your device (be sure to delete any unused steps remaining in your layers list to avoid unnecessary steps in your workflow). Simply choose a folder and give it a name, then click "Save Workflow" to record your workflow for future use. If you have already made a workflow and want to make a nother but some steps remain in the workflow log (e.g., a calculations step without an image layer to delete), click "Clear Worklow" to remove all steps from the workflow log and start over.

Below the "Save Workflow" and "Clear Workflow" widgets is the batch processing widget. The four buttons on top let you select image files (either as single files or as sequences in folders if required) to add to the batch list as well as removing any that were mistakenly added. Selected files will populate the central lighter gray area in a list for you to double check before batch processing. Below that, select a Workflow folder and an Output folder (the Workflow folder is the path to a saved workflow and the output folder is the path to where the processed images should be saved). Beneath these options are checkboxes to toggle saving every step (vs. just the final step) and whether to save images at all (in the case of only performing calculations or plots with no processing of images in the workflow).

Click "Run Batch" to process all files in the list with the provided workflow and save their outputs (images, calculations, and plots) in the Ouput folder.

![Batch Tab](/assets/images/Batch_Tab.png)

If you don't know what a widget does, simply hover over its label and any of the options within it to see a tooltip popup of what the widget does and what each parameter does. Each widget also typically provides a link to the source package's functionality so that you can learn more about what it does and how it operates should you have more questions after reading the tooltips.

### Adding Widgets

Adding your own widgets for your own custom image processing functions is very easy with PyDoug. To add your own widgets, install PyDoug through the **Developers** route described above. Then, open the `user_functions.py file` found in the `packages/pydoug-core/src/pydoug_core/user/` directory in your coding IDE of choice. In this file is a sample function (commented out) that will create a widget called "Simple Threshold" in a tab called "Users" in the PyDoug plugin. The following code is all that is required to create this widget:

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

The `@register_operation` decorator above the function is what tells PyDoug to add this function as a widget in napari. It specifies what tab to use, what the name of the widget should be, and can specify many other things such as how to handle masks, whether or not the operation should automatically be recorded in the workflow log, how to handle non-image array outputs (like plots and calculations), and more (see the **Full List of `@register_operation` Options** section below). Without this decorator, your function will not be added as a widget. Using this decorator automatically adds your function as a recordable operation that can be used in batch processing workflows without any additional code required.

In general, your function just needs to accept a numpy array (imarray) as the first parameter and return either a numpy array, numerical results (can be a single value or a dictionary of values, both are handled by PyDoug's results display widget), or a matplotlib plot (specifically must return a matplotlib.figure.Figure object, do not use pyplot and do not show the figure in the function).

The type annotations for the input parameters are what tell magicgui what type of widget input should be created (e.g., the image array creates a layer drop-down selector while the tuple creates two inputs where each is formatted using `Annotated` to only allow values in the range [0, 65535] and has default values of 50 and 200). There are many types of inputs that can be created, including toggleable checkboxes, single value floating point or integer inputs, string inputs, drop-down input selectors, slider bars, and more (see [magicgui](https://pyapp-kit.github.io/magicgui/type_map/) documentation for a comprehensive list).

The docstring at the top of the widget is what populates the tooltips automatically when the widget is created. Use the [numpy style]((https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html)) (a short description followed by `Parameters` (underlined) and then `Returns` (also underlined)) so that PyDoug can read it properly.

If there is some functionality that you want to include in the widget but are not sure how to specify this, simply look at the source code for other PyDoug widgets that use this functionality as examples. Use this sample widget as a template to get started.

### Full List of `@register_operation` Options:

```
category: str
    The category (tab) of the operation.
    Write as lowercase (it will be automatically capitalized).
    Example: category="measurements"
        - Will add this widget to "Measurements" tab.

display_name: None | str = None
    The display name of the operation (optional).
    Automatically generated if not provided.
    Example: display_name="Test Widget"
        - Will label the widget "Test Widget".

widget_options: dict | None
    (Optional) Dictionary of default widget parameter settings.
    Most options that go here can be accomplished with "Annotated".
    Example: widget_options={"Parameter_Name": {"step": 0.001, "min": 1}}
        - Changes default step of 0.01 for float to 0.001.
        - Sets a minimum value of 1.
        - Both settings applied to same "Parameter_Name" parameter.

native_options: dict[str, dict] | None
    (Optional) Dictionary of default widget native parameter settings.
    Example: native_options={"Parameter_Name": {"setDecimals": 3}}
        - Changes default decimal count for float of 2 to 3 for "Parameter_Name".

array_params: tuple[str, ...] = {}
    (Optional) Tuple of array parameters as needed for the operation.
    Use this if your widget needs to accept another image layer as a parameter.
    Example: array_params=("Mask",)
        - Creates another layer drop-down to select an image to use as a mask.

array_param_types: dict[str, str] = {}
    (Optional) The napari layer type for each array input.
    Use this if you need a non-image layer array input for any array_params.
    Example: array_param_types={"Mask": "labels"}
        - Requires "Mask" to be a parameter in array_params.
        - Lets the user select a labels layer for Mask instead of an image layer.

array_param_flags: dict[str, str]
    Relevant flags for array inputs.
    Example: array_param_flags={"Mask": "Apply_Mask"}
        - Requires "Mask" to be a parameter in array_params.
        - Requires "Apply_Mask" to be a boolean parameter in the function.
        - If Apply_Mask is False, will not record the selected Mask layer.

output_kind: Literal["image", "labels", "plot", "value"] = "image"
    Whether the operation should output an image/labels layer, plot, or value.
    Default is "image".
    Example: output_kind="plot"
        - Specifies that the operation returns a plot, not an image.

display_result: Callable[[Any], None] | None
    DO NOT USE.

save_result: Callable[[Any, Path], None] | None
    DO NOT USE.

recordable_toggle: bool = False
    Whether or not to add a checkbox to the widget to toggle recording.
    Useful for plots/calculations if to perform the operation without recording.
    Default is False.
    Example: recordable_toggle=True
        - Adds a recordable toggle for the widget.

recordable: bool
    Whether or not the operation should be recorded on use.
    Useful for operations that shouldn't be recorded in a workflow (rare).
    Example: recordable=False
        - Will not record the widget's operation in the workflow.

allow_in_place: bool
    Whether or not allow an "In_Place" checkbox to toggle in-place operations.
    In-place operations act on the layer without creating a new one.
    Useful for expensive operations (e.g., cropping a 30 GB dataset).
    IMPORTANT, cannot be undone (acts on the layer instead of making a new one).
    Up to you to code logic for "In_Place" performing the operation in place on the
    input "imarray" vs. not.
    Example: allow_in_place=True
        - Requires "In_Place" boolean parameter to be present in the parameters.
        - If "In_Place" is checked, rewrites the current layer with the new data.

live_preview: bool
    DO NOT USE.

layer_type: Literal["image", "labels"] = "image"
    The layer type for the input image array.
    Useful if a widget produces a labels layer and you want to process it later.
    Default is "image".
    Example: layer_type="labels"
        - Lets the input numpy array be from a labels layer instead of an image.

captures_viewer_settings: bool = False
    DO NOT USE.
```
