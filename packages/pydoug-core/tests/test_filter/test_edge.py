"""
Tests for PyDoug's filter.edge module.
"""

import numpy as np

from pydoug_core.filter import edge


def test_gradient_sobel(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"])
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"])
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"])
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"])
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(sample_arrays["2dgray"], Horizontal=True)
    result6: np.ndarray = edge.gradient(sample_arrays["2drgb"], Horizontal=True)
    result7: np.ndarray = edge.gradient(sample_arrays["3dgray"], Horizontal=True)
    result8: np.ndarray = edge.gradient(sample_arrays["3drgb"], Horizontal=True)
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape

    result9: np.ndarray = edge.gradient(sample_arrays["2dgray"], Vertical=True)
    result10: np.ndarray = edge.gradient(sample_arrays["2drgb"], Vertical=True)
    result11: np.ndarray = edge.gradient(sample_arrays["3dgray"], Vertical=True)
    result12: np.ndarray = edge.gradient(sample_arrays["3drgb"], Vertical=True)
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["3dgray"].shape
    assert result12.shape == sample_arrays["3drgb"].shape

    result13: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Perform_Slicewise=True
    )
    result14: np.ndarray = edge.gradient(sample_arrays["2drgb"], Perform_Slicewise=True)
    result15: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Perform_Slicewise=True
    )
    result16: np.ndarray = edge.gradient(sample_arrays["3drgb"], Perform_Slicewise=True)
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["2drgb"].shape
    assert result15.shape == sample_arrays["3dgray"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_gradient_farid(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"], Method="Farid")
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"], Method="Farid")
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"], Method="Farid")
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"], Method="Farid")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Farid", Horizontal=True
    )
    result6: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Farid", Horizontal=True
    )
    result7: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Farid", Horizontal=True
    )
    result8: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Farid", Horizontal=True
    )
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape

    result9: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Farid", Vertical=True
    )
    result10: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Farid", Vertical=True
    )
    result11: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Farid", Vertical=True
    )
    result12: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Farid", Vertical=True
    )
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["3dgray"].shape
    assert result12.shape == sample_arrays["3drgb"].shape

    result13: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Farid", Perform_Slicewise=True
    )
    result14: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Farid", Perform_Slicewise=True
    )
    result15: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Farid", Perform_Slicewise=True
    )
    result16: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Farid", Perform_Slicewise=True
    )
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["2drgb"].shape
    assert result15.shape == sample_arrays["3dgray"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_gradient_laplace(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"], Method="Laplace")
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"], Method="Laplace")
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"], Method="Laplace")
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"], Method="Laplace")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Laplace", Perform_Slicewise=True
    )
    result6: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Laplace", Perform_Slicewise=True
    )
    result7: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Laplace", Perform_Slicewise=True
    )
    result8: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Laplace", Perform_Slicewise=True
    )
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape


def test_gradient_prewitt(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"], Method="Prewitt")
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"], Method="Prewitt")
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"], Method="Prewitt")
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"], Method="Prewitt")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Prewitt", Horizontal=True
    )
    result6: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Prewitt", Horizontal=True
    )
    result7: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Prewitt", Horizontal=True
    )
    result8: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Prewitt", Horizontal=True
    )
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape

    result9: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Prewitt", Vertical=True
    )
    result10: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Prewitt", Vertical=True
    )
    result11: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Prewitt", Vertical=True
    )
    result12: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Prewitt", Vertical=True
    )
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["3dgray"].shape
    assert result12.shape == sample_arrays["3drgb"].shape

    result13: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Prewitt", Perform_Slicewise=True
    )
    result14: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Prewitt", Perform_Slicewise=True
    )
    result15: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Prewitt", Perform_Slicewise=True
    )
    result16: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Prewitt", Perform_Slicewise=True
    )
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["2drgb"].shape
    assert result15.shape == sample_arrays["3dgray"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_gradient_roberts(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"], Method="Roberts")
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"], Method="Roberts")
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"], Method="Roberts")
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"], Method="Roberts")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Roberts", Horizontal=True
    )
    result6: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Roberts", Horizontal=True
    )
    result7: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Roberts", Horizontal=True
    )
    result8: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Roberts", Horizontal=True
    )
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape

    result9: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Roberts", Vertical=True
    )
    result10: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Roberts", Vertical=True
    )
    result11: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Roberts", Vertical=True
    )
    result12: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Roberts", Vertical=True
    )
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["3dgray"].shape
    assert result12.shape == sample_arrays["3drgb"].shape

    result13: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Roberts", Perform_Slicewise=True
    )
    result14: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Roberts", Perform_Slicewise=True
    )
    result15: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Roberts", Perform_Slicewise=True
    )
    result16: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Roberts", Perform_Slicewise=True
    )
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["2drgb"].shape
    assert result15.shape == sample_arrays["3dgray"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_gradient_scharr(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"], Method="Scharr")
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"], Method="Scharr")
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"], Method="Scharr")
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"], Method="Scharr")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Scharr", Horizontal=True
    )
    result6: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Scharr", Horizontal=True
    )
    result7: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Scharr", Horizontal=True
    )
    result8: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Scharr", Horizontal=True
    )
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape

    result9: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Scharr", Vertical=True
    )
    result10: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Scharr", Vertical=True
    )
    result11: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Scharr", Vertical=True
    )
    result12: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Scharr", Vertical=True
    )
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["3dgray"].shape
    assert result12.shape == sample_arrays["3drgb"].shape

    result13: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Scharr", Perform_Slicewise=True
    )
    result14: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Scharr", Perform_Slicewise=True
    )
    result15: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Scharr", Perform_Slicewise=True
    )
    result16: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Scharr", Perform_Slicewise=True
    )
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["2drgb"].shape
    assert result15.shape == sample_arrays["3dgray"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_gradient_local(sample_arrays) -> None:

    result1: np.ndarray = edge.gradient(sample_arrays["2dgray"], Method="Local")
    result2: np.ndarray = edge.gradient(sample_arrays["2drgb"], Method="Local")
    result3: np.ndarray = edge.gradient(sample_arrays["3dgray"], Method="Local")
    result4: np.ndarray = edge.gradient(sample_arrays["3drgb"], Method="Local")
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["2drgb"].shape
    assert result3.shape == sample_arrays["3dgray"].shape
    assert result4.shape == sample_arrays["3drgb"].shape

    result5: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Local", Horizontal=True
    )
    result6: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Local", Horizontal=True
    )
    result7: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Local", Horizontal=True
    )
    result8: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Local", Horizontal=True
    )
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["2drgb"].shape
    assert result7.shape == sample_arrays["3dgray"].shape
    assert result8.shape == sample_arrays["3drgb"].shape

    result9: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Local", Vertical=True
    )
    result10: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Local", Vertical=True
    )
    result11: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Local", Vertical=True
    )
    result12: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Local", Vertical=True
    )
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["2drgb"].shape
    assert result11.shape == sample_arrays["3dgray"].shape
    assert result12.shape == sample_arrays["3drgb"].shape

    result13: np.ndarray = edge.gradient(
        sample_arrays["2dgray"], Method="Local", Perform_Slicewise=True
    )
    result14: np.ndarray = edge.gradient(
        sample_arrays["2drgb"], Method="Local", Perform_Slicewise=True
    )
    result15: np.ndarray = edge.gradient(
        sample_arrays["3dgray"], Method="Local", Perform_Slicewise=True
    )
    result16: np.ndarray = edge.gradient(
        sample_arrays["3drgb"], Method="Local", Perform_Slicewise=True
    )
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["2drgb"].shape
    assert result15.shape == sample_arrays["3dgray"].shape
    assert result16.shape == sample_arrays["3drgb"].shape


def test_ridge(sample_arrays) -> None:

    result1: np.ndarray = edge.ridge(sample_arrays["2dgray"], "Frangi")
    result2: np.ndarray = edge.ridge(sample_arrays["3dgray"], "Frangi")
    result3: np.ndarray = edge.ridge(sample_arrays["2drgb"], "Frangi")
    result4: np.ndarray = edge.ridge(sample_arrays["3drgb"], "Frangi")
    result5: np.ndarray = edge.ridge(
        sample_arrays["2dgray"], "Frangi", Perform_Slicewise=True
    )
    result6: np.ndarray = edge.ridge(
        sample_arrays["3dgray"], "Frangi", Perform_Slicewise=True
    )
    result7: np.ndarray = edge.ridge(
        sample_arrays["2drgb"], "Frangi", Perform_Slicewise=True
    )
    result8: np.ndarray = edge.ridge(
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

    result9: np.ndarray = edge.ridge(sample_arrays["2dgray"], "Hessian")
    result10: np.ndarray = edge.ridge(sample_arrays["3dgray"], "Hessian")
    result11: np.ndarray = edge.ridge(sample_arrays["2drgb"], "Hessian")
    result12: np.ndarray = edge.ridge(sample_arrays["3drgb"], "Hessian")
    result13: np.ndarray = edge.ridge(
        sample_arrays["2dgray"], "Hessian", Perform_Slicewise=True
    )
    result14: np.ndarray = edge.ridge(
        sample_arrays["3dgray"], "Hessian", Perform_Slicewise=True
    )
    result15: np.ndarray = edge.ridge(
        sample_arrays["2drgb"], "Hessian", Perform_Slicewise=True
    )
    result16: np.ndarray = edge.ridge(
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

    result17: np.ndarray = edge.ridge(sample_arrays["2dgray"], "Meijering")
    result18: np.ndarray = edge.ridge(sample_arrays["3dgray"], "Meijering")
    result19: np.ndarray = edge.ridge(sample_arrays["2drgb"], "Meijering")
    result20: np.ndarray = edge.ridge(sample_arrays["3drgb"], "Meijering")
    result21: np.ndarray = edge.ridge(
        sample_arrays["2dgray"], "Meijering", Perform_Slicewise=True
    )
    result22: np.ndarray = edge.ridge(
        sample_arrays["3dgray"], "Meijering", Perform_Slicewise=True
    )
    result23: np.ndarray = edge.ridge(
        sample_arrays["2drgb"], "Meijering", Perform_Slicewise=True
    )
    result24: np.ndarray = edge.ridge(
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

    result25: np.ndarray = edge.ridge(sample_arrays["2dgray"], "Sato")
    result26: np.ndarray = edge.ridge(sample_arrays["3dgray"], "Sato")
    result27: np.ndarray = edge.ridge(sample_arrays["2drgb"], "Sato")
    result28: np.ndarray = edge.ridge(sample_arrays["3drgb"], "Sato")
    result29: np.ndarray = edge.ridge(
        sample_arrays["2dgray"], "Sato", Perform_Slicewise=True
    )
    result30: np.ndarray = edge.ridge(
        sample_arrays["3dgray"], "Sato", Perform_Slicewise=True
    )
    result31: np.ndarray = edge.ridge(
        sample_arrays["2drgb"], "Sato", Perform_Slicewise=True
    )
    result32: np.ndarray = edge.ridge(
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

    result33: np.ndarray = edge.ridge(sample_arrays["2dgray"], "Frangi", Gamma=2)
    result34: np.ndarray = edge.ridge(sample_arrays["2dgray"], "Hessian", Gamma=2)
    assert result33.dtype == np.dtype(np.float64)
    assert result34.dtype == np.dtype(np.float64)
    assert result33.shape == sample_arrays["2dgray"].shape
    assert result34.shape == sample_arrays["2dgray"].shape
