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
