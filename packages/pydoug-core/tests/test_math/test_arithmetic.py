"""
Tests for PyDoug's math.arithmetic module.
"""

import numpy as np

from pydoug_core.math import arithmetic


def test_im_arithmetic(sample_3dgray_array) -> None:

    math_array: np.ndarray = np.copy(sample_3dgray_array)
    math_array[math_array == 0] = 1
    bool_array: np.ndarray = math_array >= 128
    float_array: np.ndarray = math_array.astype(np.float64)
    result1: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, Use_Arithmetic_Layer=True, Arithmetic_Layer=math_array
    )
    result2: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        "Divide",
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
    )
    result3: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        "Multiply",
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
    )
    result4: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        "Subtract",
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
    )
    result5: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
        Preserve_Dtype=False,
    )
    result6: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        "Divide",
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
        Preserve_Dtype=False,
    )
    result7: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        "Multiply",
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
        Preserve_Dtype=False,
    )
    result8: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array,
        "Subtract",
        Use_Arithmetic_Layer=True,
        Arithmetic_Layer=math_array,
        Preserve_Dtype=False,
    )
    result9: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, Arithmetic_Value=128
    )
    result10: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, "Divide", Arithmetic_Value=128
    )
    result11: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, "Multiply", Arithmetic_Value=128
    )
    result12: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, "Subtract", Arithmetic_Value=128
    )
    result13: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, Arithmetic_Value=128, Preserve_Dtype=False
    )
    result14: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, "Divide", Arithmetic_Value=128, Preserve_Dtype=False
    )
    result15: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, "Multiply", Arithmetic_Value=128, Preserve_Dtype=False
    )
    result16: np.ndarray = arithmetic.arithmetic(
        sample_3dgray_array, "Subtract", Arithmetic_Value=128, Preserve_Dtype=False
    )
    result17: np.ndarray = arithmetic.arithmetic(bool_array)
    result18: np.ndarray = arithmetic.arithmetic(float_array)

    assert result1.dtype == sample_3dgray_array.dtype
    assert result2.dtype == sample_3dgray_array.dtype
    assert result3.dtype == sample_3dgray_array.dtype
    assert result4.dtype == sample_3dgray_array.dtype
    assert result5.dtype != sample_3dgray_array.dtype
    assert result6.dtype != sample_3dgray_array.dtype
    assert result7.dtype != sample_3dgray_array.dtype
    assert result8.dtype != sample_3dgray_array.dtype
    assert result9.dtype == sample_3dgray_array.dtype
    assert result10.dtype == sample_3dgray_array.dtype
    assert result11.dtype == sample_3dgray_array.dtype
    assert result12.dtype == sample_3dgray_array.dtype
    assert result13.dtype != sample_3dgray_array.dtype
    assert result14.dtype != sample_3dgray_array.dtype
    assert result15.dtype != sample_3dgray_array.dtype
    assert result16.dtype != sample_3dgray_array.dtype
    assert result17.dtype == bool_array.dtype
    assert result18.dtype == float_array.dtype
