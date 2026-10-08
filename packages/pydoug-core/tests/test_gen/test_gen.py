"""
Tests for PyDoug's gen.gen module.
"""

import numpy as np

from pydoug_core.gen import gen


def test_gen_structres() -> None:

    result1: np.ndarray = gen.gen_structures(Dimensions=(25, 25, 0))
    result2: np.ndarray = gen.gen_structures(Dimensions=(25, 25, 25))
    result5: np.ndarray = gen.gen_structures("Lattice Spheres", (25, 25, 0))
    result6: np.ndarray = gen.gen_structures("Lattice Spheres", (25, 25, 25))
    result7: np.ndarray = gen.gen_structures(
        "Lattice Spheres", (25, 25, 0), Lattice="BCC"
    )
    result8: np.ndarray = gen.gen_structures(
        "Lattice Spheres", (25, 25, 25), Lattice="BCC"
    )
    result9: np.ndarray = gen.gen_structures(
        "Lattice Spheres", (25, 25, 0), Lattice="FCC"
    )
    result10: np.ndarray = gen.gen_structures(
        "Lattice Spheres", (25, 25, 25), Lattice="FCC"
    )
    result11: np.ndarray = gen.gen_structures(
        "Lattice Spheres", (25, 25, 0), Lattice="Triangular"
    )
    result12: np.ndarray = gen.gen_structures(
        "Lattice Spheres", (25, 25, 25), Lattice="Triangular"
    )
    result13: np.ndarray = gen.gen_structures("Overlapping Spheres", (25, 25, 0))
    result14: np.ndarray = gen.gen_structures("Overlapping Spheres", (25, 25, 25))
    result15: np.ndarray = gen.gen_structures("Polydisperse Spheres", (25, 25, 0))
    result16: np.ndarray = gen.gen_structures("Polydisperse Spheres", (25, 25, 25))
    result17: np.ndarray = gen.gen_structures("Random Spheres", (25, 25, 0))
    result18: np.ndarray = gen.gen_structures("Random Spheres", (25, 25, 25))

    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result5.dtype == np.bool
    assert result6.dtype == np.bool
    assert result7.dtype == np.bool
    assert result8.dtype == np.bool
    assert result9.dtype == np.bool
    assert result10.dtype == np.bool
    assert result11.dtype == np.bool
    assert result12.dtype == np.bool
    assert result13.dtype == np.bool
    assert result14.dtype == np.bool
    assert result15.dtype == np.bool
    assert result16.dtype == np.bool
    assert result17.dtype == np.bool
    assert result18.dtype == np.bool
    assert result1.shape == (25, 25)
    assert result2.shape == (25, 25, 25)
    assert result5.shape == (25, 25)
    assert result6.shape == (25, 25, 25)
    assert result7.shape == (25, 25, 25)
    assert result8.shape == (25, 25, 25)
    assert result9.shape == (25, 25, 25)
    assert result10.shape == (25, 25, 25)
    assert result11.shape == (25, 25)
    assert result12.shape == (25, 25)
    assert result13.shape == (25, 25)
    assert result14.shape == (25, 25, 25)
    assert result15.shape == (25, 25)
    assert result16.shape == (25, 25, 25)
    assert result17.shape == (25, 25)
    assert result18.shape == (25, 25, 25)
