"""
Tests for PyDoug's filter.ridge module.
"""

import numpy as np

from pydoug_core.filter import ridge


def test_ridge(sample_arrays) -> None:

    result1: np.ndarray = ridge.ridge(sample_arrays["2dgray"], "Frangi")
    result2: np.ndarray = ridge.ridge(sample_arrays["3dgray"], "Frangi")
    result3: np.ndarray = ridge.ridge(sample_arrays["2drgb"], "Frangi")
    result4: np.ndarray = ridge.ridge(sample_arrays["3drgb"], "Frangi")
    result5: np.ndarray = ridge.ridge(
        sample_arrays["2dgray"], "Frangi", Perform_Slicewise=True
    )
    result6: np.ndarray = ridge.ridge(
        sample_arrays["3dgray"], "Frangi", Perform_Slicewise=True
    )
    result7: np.ndarray = ridge.ridge(
        sample_arrays["2drgb"], "Frangi", Perform_Slicewise=True
    )
    result8: np.ndarray = ridge.ridge(
        sample_arrays["3drgb"], "Frangi", Perform_Slicewise=True
    )
    assert result1.dtype == np.dtype(np.float64)
    assert result2.dtype == np.dtype(np.float64)
    assert result3.dtype == np.dtype(np.float64)
    assert result4.dtype == np.dtype(np.float64)
    assert result5.dtype == np.dtype(np.float64)
    assert result6.dtype == np.dtype(np.float64)
    assert result7.dtype == np.dtype(np.float64)
    assert result8.dtype == np.dtype(np.float64)
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["2drgb"].shape
    assert result8.shape == sample_arrays["3drgb"].shape
    assert np.all(result1 == result5)
    assert np.all(result3 == result7)

    result9: np.ndarray = ridge.ridge(sample_arrays["2dgray"], "Hessian")
    result10: np.ndarray = ridge.ridge(sample_arrays["3dgray"], "Hessian")
    result11: np.ndarray = ridge.ridge(sample_arrays["2drgb"], "Hessian")
    result12: np.ndarray = ridge.ridge(sample_arrays["3drgb"], "Hessian")
    result13: np.ndarray = ridge.ridge(
        sample_arrays["2dgray"], "Hessian", Perform_Slicewise=True
    )
    result14: np.ndarray = ridge.ridge(
        sample_arrays["3dgray"], "Hessian", Perform_Slicewise=True
    )
    result15: np.ndarray = ridge.ridge(
        sample_arrays["2drgb"], "Hessian", Perform_Slicewise=True
    )
    result16: np.ndarray = ridge.ridge(
        sample_arrays["3drgb"], "Hessian", Perform_Slicewise=True
    )
    assert result9.dtype == np.dtype(np.float64)
    assert result10.dtype == np.dtype(np.float64)
    assert result11.dtype == np.dtype(np.float64)
    assert result12.dtype == np.dtype(np.float64)
    assert result13.dtype == np.dtype(np.float64)
    assert result14.dtype == np.dtype(np.float64)
    assert result15.dtype == np.dtype(np.float64)
    assert result16.dtype == np.dtype(np.float64)
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["3dgray"].shape
    assert result11.shape == sample_arrays["2drgb"].shape
    assert result12.shape == sample_arrays["3drgb"].shape
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["3dgray"].shape
    assert result15.shape == sample_arrays["2drgb"].shape
    assert result16.shape == sample_arrays["3drgb"].shape
    assert np.all(result9 == result13)
    assert np.all(result11 == result15)

    result17: np.ndarray = ridge.ridge(sample_arrays["2dgray"], "Meijering")
    result18: np.ndarray = ridge.ridge(sample_arrays["3dgray"], "Meijering")
    result19: np.ndarray = ridge.ridge(sample_arrays["2drgb"], "Meijering")
    result20: np.ndarray = ridge.ridge(sample_arrays["3drgb"], "Meijering")
    result21: np.ndarray = ridge.ridge(
        sample_arrays["2dgray"], "Meijering", Perform_Slicewise=True
    )
    result22: np.ndarray = ridge.ridge(
        sample_arrays["3dgray"], "Meijering", Perform_Slicewise=True
    )
    result23: np.ndarray = ridge.ridge(
        sample_arrays["2drgb"], "Meijering", Perform_Slicewise=True
    )
    result24: np.ndarray = ridge.ridge(
        sample_arrays["3drgb"], "Meijering", Perform_Slicewise=True
    )
    assert result17.dtype == np.dtype(np.float64)
    assert result18.dtype == np.dtype(np.float64)
    assert result19.dtype == np.dtype(np.float64)
    assert result20.dtype == np.dtype(np.float64)
    assert result21.dtype == np.dtype(np.float64)
    assert result22.dtype == np.dtype(np.float64)
    assert result23.dtype == np.dtype(np.float64)
    assert result24.dtype == np.dtype(np.float64)
    assert result17.shape == sample_arrays["2dgray"].shape
    assert result18.shape == sample_arrays["3dgray"].shape
    assert result19.shape == sample_arrays["2drgb"].shape
    assert result20.shape == sample_arrays["3drgb"].shape
    assert result21.shape == sample_arrays["2dgray"].shape
    assert result22.shape == sample_arrays["3dgray"].shape
    assert result23.shape == sample_arrays["2drgb"].shape
    assert result24.shape == sample_arrays["3drgb"].shape
    assert np.all(result17 == result21)
    assert np.all(result19 == result23)

    result25: np.ndarray = ridge.ridge(sample_arrays["2dgray"], "Sato")
    result26: np.ndarray = ridge.ridge(sample_arrays["3dgray"], "Sato")
    result27: np.ndarray = ridge.ridge(sample_arrays["2drgb"], "Sato")
    result28: np.ndarray = ridge.ridge(sample_arrays["3drgb"], "Sato")
    result29: np.ndarray = ridge.ridge(
        sample_arrays["2dgray"], "Sato", Perform_Slicewise=True
    )
    result30: np.ndarray = ridge.ridge(
        sample_arrays["3dgray"], "Sato", Perform_Slicewise=True
    )
    result31: np.ndarray = ridge.ridge(
        sample_arrays["2drgb"], "Sato", Perform_Slicewise=True
    )
    result32: np.ndarray = ridge.ridge(
        sample_arrays["3drgb"], "Sato", Perform_Slicewise=True
    )
    assert result25.dtype == np.dtype(np.float64)
    assert result26.dtype == np.dtype(np.float64)
    assert result27.dtype == np.dtype(np.float64)
    assert result28.dtype == np.dtype(np.float64)
    assert result29.dtype == np.dtype(np.float64)
    assert result30.dtype == np.dtype(np.float64)
    assert result31.dtype == np.dtype(np.float64)
    assert result32.dtype == np.dtype(np.float64)
    assert result25.shape == sample_arrays["2dgray"].shape
    assert result26.shape == sample_arrays["3dgray"].shape
    assert result27.shape == sample_arrays["2drgb"].shape
    assert result28.shape == sample_arrays["3drgb"].shape
    assert result29.shape == sample_arrays["2dgray"].shape
    assert result30.shape == sample_arrays["3dgray"].shape
    assert result31.shape == sample_arrays["2drgb"].shape
    assert result32.shape == sample_arrays["3drgb"].shape
    assert np.all(result25 == result29)
    assert np.all(result27 == result31)

    result33: np.ndarray = ridge.ridge(sample_arrays["2dgray"], "Frangi", Gamma=2)
    result34: np.ndarray = ridge.ridge(sample_arrays["2dgray"], "Hessian", Gamma=2)
    assert result33.dtype == np.dtype(np.float64)
    assert result34.dtype == np.dtype(np.float64)
    assert result33.shape == sample_arrays["2dgray"].shape
    assert result34.shape == sample_arrays["2dgray"].shape
