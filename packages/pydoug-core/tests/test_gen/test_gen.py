"""
Tests for PyDoug's gen.gen module.
"""

import numpy as np
from skimage import data

from pydoug_core.gen import gen


def test_blank_image() -> None:

    result1: np.ndarray = gen.blank_image((10, 10, 0))
    result2: np.ndarray = gen.blank_image((10, 10, 0), "Float32")
    result3: np.ndarray = gen.blank_image((10, 10, 0), "Bool")
    result4: np.ndarray = gen.blank_image((10, 10, 10))
    result5: np.ndarray = gen.blank_image((10, 10, 0), Value=300)
    result6: np.ndarray = gen.blank_image((10, 10, 0), Value=-50)
    result7: np.ndarray = gen.blank_image((10, 10, 0), "Uint16")
    result8: np.ndarray = gen.blank_image((10, 10, 0), "Int16")
    result9: np.ndarray = gen.blank_image((10, 10, 0), "Float64")

    assert result1.dtype == np.dtype(np.uint8)
    assert result2.dtype == np.dtype(np.float32)
    assert result3.dtype == np.dtype(np.bool)
    assert result4.dtype == np.dtype(np.uint8)
    assert result5.dtype == np.dtype(np.uint8)
    assert result6.dtype == np.dtype(np.uint8)
    assert result7.dtype == np.dtype(np.uint16)
    assert result8.dtype == np.dtype(np.int16)
    assert result9.dtype == np.dtype(np.float64)
    assert np.all(result1 == 0)
    assert np.all(result2 == 0)
    assert np.all(result3 == 0)
    assert np.all(result4 == 0)
    assert np.all(result5 == 255)
    assert np.all(result6 == 0)
    assert np.all(result7 == 0)
    assert np.all(result8 == 0)
    assert np.all(result9 == 0)
    assert result1.shape == (10, 10)
    assert result2.shape == (10, 10)
    assert result3.shape == (10, 10)
    assert result4.shape == (10, 10, 10)
    assert result5.shape == (10, 10)
    assert result6.shape == (10, 10)
    assert result7.shape == (10, 10)
    assert result8.shape == (10, 10)
    assert result9.shape == (10, 10)


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


def test_scikit_data() -> None:

    assert np.all(gen.scikit_data()[0:10, 0:10] == data.hubble_deep_field()[0:10, 0:10])
    assert np.all(gen.scikit_data("Brain")[0:10, 0:10] == data.brain()[0:10, 0:10])
    assert np.all(gen.scikit_data("Cell")[0:10, 0:10] == data.cell()[0:10, 0:10])
    assert np.all(gen.scikit_data("Cells3D")[0:10, 0:10] == data.cells3d()[0:10, 0:10])
    assert np.all(
        gen.scikit_data("Immunohistochemistry")[0:10, 0:10]
        == data.immunohistochemistry()[0:10, 0:10]
    )
    assert np.all(gen.scikit_data("Kidney")[0:10, 0:10] == data.kidney()[0:10, 0:10])
    assert np.all(gen.scikit_data("Lily")[0:10, 0:10] == data.lily()[0:10, 0:10])
    assert np.all(
        gen.scikit_data("Microaneurysms")[0:10, 0:10]
        == data.microaneurysms()[0:10, 0:10]
    )
    assert np.all(gen.scikit_data("Moon")[0:10, 0:10] == data.moon()[0:10, 0:10])
    assert np.all(
        gen.scikit_data("Nickel Solidification")[0:10, 0:10]
        == data.nickel_solidification()[0:10, 0:10]
    )
    assert np.all(
        gen.scikit_data("Protein Transport")[0:10, 0:10]
        == data.protein_transport()[0:10, 0:10]
    )
    assert np.all(gen.scikit_data("Retina")[0:10, 0:10] == data.retina()[0:10, 0:10])
    assert np.all(
        gen.scikit_data("Shepp Logan Phantom")[0:10, 0:10]
        == data.shepp_logan_phantom()[0:10, 0:10]
    )
    assert np.all(gen.scikit_data("Skin")[0:10, 0:10] == data.skin()[0:10, 0:10])
