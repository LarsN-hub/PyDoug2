"""
Generate image data.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["blank_image", "gen_structures", "scikit_data"]


@register_operation(
    category="generate",
    display_name="Blank Image",
    requires_layer=False,
    recordable=False,
)
def blank_image(
    Dimensions: tuple[
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 0}],
    ] = (100, 100, 0),
    Dtype: Literal["Uint8", "Uint16", "Int16", "Float32", "Float64", "Bool"] = "Uint8",
    Value: Annotated[float, {"step": 0.0001}] = 0,
) -> np.ndarray:
    """
    Generate a blank image of a specified dtype filled with a single value. Covnenience
    wrapper around numpy's empty:
    https://numpy.org/doc/stable/reference/generated/numpy.empty.html

    Parameters
    ----------
    Dimensions: tuple[int, int, int]
        The dimensions of the image in the format [X, Y, Z]. If Z is 0, a 2D image is
        returned.
    Dtype: Literal
        The data type of the image. Default is "Uint8". Note, the range for floating
        point images is large, but most operations with floating point images assume a
        range of [0, 1].
        - Uint8: 8-bit unsigned integer format [0, 255]
        - Uint16: 16-bit unsigned integer format [0, 65,535]
        - Int16: 16-bit signed integer format [-32,768, 32,767]
        - Float32: 32-bit (single-precision) floating-point format [-3.4e+38, 3.4e+38]
        - Float64: 64-bit (double-precision) floating-point format [-1.8e+308, 1.8e+308]
        - Bool: Boolean True/False format
    Value: float
        The value to populate the blank image with. If the value is outside the dtype's
        range it will be clipped to the nearest boundary. If an integer data type is
        chosen and a decimal value is provided, it will be round to the nearest integer.

    Returns
    -------
    array: np.ndarray
        The blank image.
    """

    from pydoug_core.util import str_2_dtype

    if Dtype in {"Uint8", "Uint16", "Int16"}:
        dtype_max: bool | float | int = np.iinfo(str_2_dtype(Dtype)).max
        dtype_min: bool | float | int = np.iinfo(str_2_dtype(Dtype)).min
    elif Dtype in {"Float32", "Float64"}:
        dtype_max = np.finfo(str_2_dtype(Dtype)).max
        dtype_min = np.finfo(str_2_dtype(Dtype)).min
    else:
        dtype_max = 1
        dtype_min = 0

    if Value > dtype_max:
        Value = dtype_max
    elif Value < dtype_min:
        Value = dtype_min

    if Dtype in {"Bool", "Uint8", "Uint16", "Int16"}:
        Value = round(Value)

    if Dimensions[2] == 0:
        dims: tuple = (Dimensions[1], Dimensions[0])
    else:
        dims: tuple = (Dimensions[2], Dimensions[1], Dimensions[0])

    array = np.empty(dims, str_2_dtype(Dtype))
    array[...] = Value
    return array


@register_operation(
    category="generate",
    display_name="Random Structures",
    requires_layer=False,
    recordable=False,
)
def gen_structures(
    Method: Literal[
        "Blobs",
        "Lattice Spheres",
        "Overlapping Spheres",
        "Polydisperse Spheres",
        "Random Spheres",
    ] = "Blobs",
    Dimensions: tuple[
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 1}],
        Annotated[int, {"min": 0}],
    ] = (100, 100, 0),
    Porosity: Annotated[float, {"min": 0, "max": 1}] = 0.5,
    Blobiness: Annotated[int, {"min": 1}] = 1,
    Radius: Annotated[int, {"min": 1}] = 5,
    Standard_Deviation: Annotated[int, {"min": 1}] = 2,
    Lattice: Literal["SC", "Triangular", "FCC", "BCC"] = "SC",
) -> np.ndarray:
    """
    Generate an image of specified dimensions with randomized structures. Convenience
    wrapper around functions from porespy's generators package, namely blobs,
    lattice_spheres, overlapping_spheres, polydisperse_spheres, and random_spheres:
    https://porespy.org/autoapi/porespy/generators/blobs.html
    https://porespy.org/autoapi/porespy/generators/lattice_spheres.html
    https://porespy.org/autoapi/porespy/generators/overlapping_spheres.html
    https://porespy.org/autoapi/porespy/generators/polydisperse_spheres.html
    https://porespy.org/autoapi/porespy/generators/random_spheres.html

    Parameters
    ----------
    Method: Literal
        The method for structure generation.
        - "Blobs": Generates blobs. 2D/3D.
        - "Lattice Spheres": Generates spheres in a lattice. 2D/3D depending on lattice.
        - "Overlapping Spheres" Generates overlapping monodisperse spheres. 2D/3D.
        - "Polydisperse Spheres": Generates overlapping polydisperse spheres. 2D/3D.
        - "Random Spheres": Generates spheres with Turquato random packing. 2D/3D.
    Dimensions: tuple[int, int int]
        The dimensions of the output image in the format [X, Y, Z].
    Porosity: float
        If "Method" is "Blobs", "Overlapping Spheres", "Polydisperse Spheres", or
        "Random Spheres", the porosity to use.
    Blobiness: int
        If "Method" is "Blobs", an integer indicating the blob morphology. Higher =
        more, smaller blobs.
    Radius: int
        If "Method" is "Lattice Spheres", "Overlapping Spheres", "Polydisperse Spheres",
        or "Random Spheres", the radius to use for shape generation. This is the mean
        radius for "Polydisperse Spheres".
    Standard_Deviation: int
        If "Method" is "Polydisperse Spheres", the standard deviation of the radius.
    Lattice: Literal
        If "Method" is "Lattice Spheres", the lattice to use. Default is "SC".
        - "BCC": Body-centered cubic. 3D only.
        - "FCC": Face-centered cubic. 3D only.
        - "SC": Simple cubic. 2D/3D.
        - "Triangular": 2D only.

    Returns
    -------
    gen_array: np.ndarray
        The generated image.
    """

    from porespy.generators import (
        blobs,
        lattice_spheres,
        overlapping_spheres,
        polydisperse_spheres,
        random_spheres,
    )
    from scipy.stats import norm

    if Dimensions[-1] == 0:
        dims: list[int] = [Dimensions[1], Dimensions[0]]
    else:
        dims = [Dimensions[2], Dimensions[1], Dimensions[0]]

    if Method == "Blobs":
        return blobs(dims, Porosity, Blobiness)

    elif Method == "Lattice Spheres":
        if Lattice == "BCC":
            latt: str = "bcc"
            if len(dims) == 2:
                dims.append(min(dims))
        elif Lattice == "FCC":
            latt = "fcc"
            if len(dims) == 2:
                dims.append(min(dims))
        elif Lattice == "SC":
            latt = "sc"
        else:
            latt = "tri"
            if len(dims) == 3:
                dims = dims[:-1]
        return lattice_spheres(dims, Radius, lattice=latt)

    elif Method == "Overlapping Spheres":
        return overlapping_spheres(dims, Radius, Porosity)

    elif Method == "Polydisperse Spheres":
        return polydisperse_spheres(
            dims, Porosity, norm(loc=Radius, scale=Standard_Deviation)
        )

    return random_spheres(dims, r=Radius, phi=Porosity, edges="extended")


@register_operation(
    category="generate",
    display_name="Scikit-Image Data",
    requires_layer=False,
    recordable=False,
)
def scikit_data(
    Data: Literal[
        "Brain",
        "Cell",
        "Cells3D",
        "Hubble Deep Field",
        "Immunohistochemistry",
        "Kidney",
        "Lily",
        "Microaneurysms",
        "Moon",
        "Nickel Solidification",
        "Protein Transport",
        "Retina",
        "Shepp Logan Phantom",
        "Skin",
    ] = "Hubble Deep Field",
) -> np.ndarray:
    """
    Generate a sample data image from scikit-image's data package:
    https://scikit-image.org/docs/stable/api/skimage.data.html#

    Parameters
    ----------
    Data: Literal
        The data option.

    Returns
    -------
    data_array: np.ndarray
        The data image.
    """

    if Data == "Brain":
        from skimage.data import brain

        return brain()
    elif Data == "Cell":
        from skimage.data import cell

        return cell()
    elif Data == "Cells3D":
        from skimage.data import cells3d

        return cells3d()
    elif Data == "Immunohistochemistry":
        from skimage.data import immunohistochemistry

        return immunohistochemistry()
    elif Data == "Kidney":
        from skimage.data import kidney

        return kidney()
    elif Data == "Lily":
        from skimage.data import lily

        return lily()
    elif Data == "Microaneurysms":
        from skimage.data import microaneurysms

        return microaneurysms()
    elif Data == "Moon":
        from skimage.data import moon

        return moon()
    elif Data == "Nickel Solidification":
        from skimage.data import nickel_solidification

        return nickel_solidification()
    elif Data == "Protein Transport":
        from skimage.data import protein_transport

        return protein_transport()
    elif Data == "Retina":
        from skimage.data import retina

        return retina()
    elif Data == "Shepp Logan Phantom":
        from skimage.data import shepp_logan_phantom

        return shepp_logan_phantom()
    elif Data == "Skin":
        from skimage.data import skin

        return skin()
    else:
        from skimage.data import hubble_deep_field

        return hubble_deep_field()
