"""
Generate image data.
"""

from typing import Annotated, Literal

from numpy import ndarray
from porespy.generators import (
    blobs,
    cylinders,
    lattice_spheres,
    overlapping_spheres,
    polydisperse_spheres,
    random_spheres,
)
from scipy.stats import norm

from pydoug_core.registry import register_operation

__all__ = ["gen_structures"]


@register_operation(
    category="generate", display_name="Random Structures", requires_layer=False
)
def gen_structures(
    Method: Literal[
        "Blobs",
        "Cylinders",
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
) -> ndarray:
    """
    Generate an image of specified dimensions with randomized structures. Convenience
    wrapper around functions from porespy's generators package, namely blobs, cylinders,
    lattice_spheres, overlapping_spheres, polydisperse_spheres, and random_spheres:
    https://porespy.org/autoapi/porespy/generators/blobs.html
    https://porespy.org/autoapi/porespy/generators/cylinders.html
    https://porespy.org/autoapi/porespy/generators/lattice_spheres.html
    https://porespy.org/autoapi/porespy/generators/overlapping_spheres.html
    https://porespy.org/autoapi/porespy/generators/polydisperse_spheres.html
    https://porespy.org/autoapi/porespy/generators/random_spheres.html

    Parameters
    ----------
    Method: Literal
        The method for structure generation.
        - "Blobs": Generates blobs. 2D/3D.
        - "Cylinders": Generates cylinders. Must be 3D.
        - "Lattice Spheres": Generates spheres in a lattice. 2D/3D depending on lattice.
        - "Overlapping Spheres" Generates overlapping monodisperse spheres. 2D/3D.
        - "Polydisperse Spheres": Generates overlapping polydisperse spheres. 2D/3D.
        - "Random Spheres": Generates spheres with Turquato random packing. 2D/3D.
    Dimensions: tuple[int, int int]
        The dimensions of the output image in the format [X, Y, Z].
    Porosity: float
        If "Method" is "Blobs", "Cylinders", "Overlapping Spheres", "Polydisperse
        Spheres", or "Random Spheres", the porosity to use.
    Blobiness: int
        If "Method" is "Blobs", an integer indicating the blob morphology. Higher =
        more, smaller blobs.
    Radius: int
        If "Method" is "Cylinders", "Lattice Spheres", "Overlapping Spheres",
        "Polydisperse Spheres", or "Random Spheres", the radius to use for shape
        generation. This is the mean radius for "Polydisperse Spheres".
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

    if Dimensions[-1] == 0:
        dims: list[int] = list(Dimensions[:-1])
    else:
        dims = list(Dimensions)

    if Method == "Blobs":
        return blobs(dims, Porosity, Blobiness)

    elif Method == "Cylinders":
        if len(dims) == 2:
            dims.append(min(dims))
        return cylinders(dims, Radius, porosity=Porosity)

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
