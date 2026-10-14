"""
Tests for PyDoug's analyze.image_info module.
"""

from typing import Any

import numpy as np

from pydoug_core.analyze import image_info
from pydoug_core.results import PlotResult


def test_stats(sample_arrays, sample_masks) -> None:

    blank_mask: np.ndarray = np.zeros(sample_arrays["2dgray"].shape, np.bool)
    result0 = image_info.stats(sample_arrays["2dgray"], True, blank_mask)
    result1 = image_info.stats(sample_arrays["2dgray"])
    result2 = image_info.stats(sample_arrays["2dgray"], True, sample_masks["2dgray"])
    result3 = image_info.stats(np.astype(sample_arrays["2dgray"], np.float64))
    result4 = image_info.stats(sample_masks["2dgray"])
    result5 = image_info.stats(sample_masks["2dgray"], True, sample_masks["2dgray"])
    result6 = image_info.stats(sample_arrays["2dgray"], Perform_Slicewise=True)
    result7 = image_info.stats(sample_masks["3dgray"], Perform_Slicewise=True)
    result8 = image_info.stats(
        sample_masks["3dgray"], True, sample_masks["3dgray"], True
    )

    assert isinstance(result0, dict)
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert isinstance(result3, dict)
    assert isinstance(result4, dict)
    assert isinstance(result5, dict)
    assert isinstance(result6, dict)
    assert isinstance(result7, PlotResult)
    assert isinstance(result8, PlotResult)

    assert result0["Mean"] is np.nan
    assert result0["Median"] is np.nan
    assert result0["Mode"] is np.nan
    assert result0["Std Dev"] is np.nan
    assert result0["Variance"] is np.nan
    assert result0["Count"] == 0
    assert result0["Min"] is np.nan
    assert result0["Max"] is np.nan
    assert result0["Dtype Min"] == 0
    assert result0["Dtype Max"] == 255

    assert isinstance(result1["Mean"], float)
    assert isinstance(result1["Median"], float)
    assert not isinstance(result1["Mode"], float) and not isinstance(
        result1["Mode"], int
    )
    assert result1["Mode"].dtype == np.dtype(np.uint8)
    assert isinstance(result1["Std Dev"], float)
    assert isinstance(result1["Variance"], float)
    assert isinstance(result1["Count"], int)
    assert not isinstance(result1["Min"], float) and not isinstance(result1["Min"], int)
    assert not isinstance(result1["Max"], float) and not isinstance(result1["Max"], int)
    assert not isinstance(result1["Range"], float) and not isinstance(
        result1["Range"], int
    )
    assert result1["Dtype Min"] == 0
    assert result1["Dtype Max"] == 255
    assert result1 == result6

    assert isinstance(result2["Mean"], float)
    assert isinstance(result2["Median"], float)
    assert not isinstance(result2["Mode"], float) and not isinstance(
        result2["Mode"], int
    )
    assert result2["Mode"].dtype == np.dtype(np.uint8)
    assert isinstance(result2["Std Dev"], float)
    assert isinstance(result2["Variance"], float)
    assert isinstance(result2["Count"], int)
    assert not isinstance(result2["Min"], float) and not isinstance(result2["Min"], int)
    assert not isinstance(result2["Max"], float) and not isinstance(result2["Max"], int)
    assert not isinstance(result2["Range"], float) and not isinstance(
        result2["Range"], int
    )
    assert result2["Dtype Min"] == 0
    assert result2["Dtype Max"] == 255

    assert isinstance(result3["Mean"], float)
    assert isinstance(result3["Median"], float)
    assert isinstance(result3["Mode"], float)
    assert isinstance(result3["Std Dev"], float)
    assert isinstance(result3["Variance"], float)
    assert isinstance(result3["Count"], int)
    assert isinstance(result3["Min"], float)
    assert isinstance(result3["Max"], float)
    assert isinstance(result3["Range"], float)
    assert result3["Dtype Min"] == np.finfo(np.float64).min
    assert result3["Dtype Max"] == np.finfo(np.float64).max

    assert isinstance(result4["Mean"], float)
    assert isinstance(result4["Median"], float)
    assert isinstance(result4["Mode"], int)
    assert isinstance(result4["Std Dev"], float)
    assert isinstance(result4["Variance"], float)
    assert isinstance(result4["Count"], int)
    assert isinstance(result4["Min"], np.bool)
    assert isinstance(result4["Max"], np.bool)
    assert isinstance(result4["Range"], int)
    assert result4["Dtype Min"] == 0
    assert result4["Dtype Max"] == 1

    assert isinstance(result5["Mean"], float)
    assert isinstance(result5["Median"], float)
    assert isinstance(result5["Mode"], int)
    assert isinstance(result5["Std Dev"], float)
    assert isinstance(result5["Variance"], float)
    assert isinstance(result5["Count"], int)
    assert isinstance(result5["Min"], np.bool)
    assert isinstance(result5["Max"], np.bool)
    assert isinstance(result5["Range"], int)
    assert result5["Dtype Min"] == 0
    assert result5["Dtype Max"] == 1


def test_linescan(sample_arrays) -> None:

    class DummyShapes:
        def __init__(self, data: list[np.ndarray], shape_type: list[str]):
            self.data = data
            self.shape_type = shape_type

    shapes_layer0: DummyShapes = DummyShapes([np.array([[1, 2], [3, 4]])], ["ellipse"])
    shapes_layer1: DummyShapes = DummyShapes([np.array([[1, 2], [3, 4]])], ["line"])
    shapes_layer2: DummyShapes = DummyShapes(
        [np.array([[3, 1, 2], [3, 3, 4]])], ["line"]
    )
    shapes_layer3: DummyShapes = DummyShapes(
        [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])], ["ellipse", "line"]
    )

    assert image_info.linescan(shapes_layer0, sample_arrays["2dgray"]) is None
    assert isinstance(
        image_info.linescan(shapes_layer1, sample_arrays["2dgray"]), PlotResult
    )
    assert isinstance(
        image_info.linescan(shapes_layer2, sample_arrays["3dgray"]), PlotResult
    )
    assert isinstance(
        image_info.linescan(shapes_layer3, sample_arrays["2dgray"]), PlotResult
    )
    assert isinstance(
        image_info.linescan(shapes_layer2, sample_arrays["2dgray"]), PlotResult
    )
    assert isinstance(
        image_info.linescan(shapes_layer1, sample_arrays["3dgray"]), PlotResult
    )


def test_percentile(sample_2dgray_array) -> None:

    result1: dict = image_info.percentiles(sample_2dgray_array)
    result2: dict = image_info.percentiles(sample_2dgray_array, (30, 70))
    result3: dict = image_info.percentiles(sample_2dgray_array, (-5, 105))
    result4: dict = image_info.percentiles(sample_2dgray_array, (0, 100))
    assert list(result1.values())[0] < list(result2.values())[0]
    assert list(result1.values())[1] > list(result2.values())[1]
    assert result4 == result3


def test_thresholds(sample_3dgray_array, sample_masks) -> None:

    mod_array1: np.ndarray = np.random.binomial(20, 0.5, (5, 10, 10)).astype(np.uint8)
    mod_array2: np.ndarray = 255 - mod_array1
    min_array: np.ndarray = np.stack((mod_array1, mod_array2), 0)
    result1: Any = image_info.thresholds(sample_3dgray_array)
    result2: Any = image_info.thresholds(sample_3dgray_array, "Isodata")
    result3: Any = image_info.thresholds(sample_3dgray_array, "Li")
    result4: Any = image_info.thresholds(sample_3dgray_array, "Mean")
    result5: Any = image_info.thresholds(min_array, "Minimum")
    result6: Any = image_info.thresholds(sample_3dgray_array, "Triangle")
    result7: Any = image_info.thresholds(sample_3dgray_array, "Yen")
    result8: Any = image_info.thresholds(sample_3dgray_array, "Otsu", Otsu_Classes=3)
    result9: Any = image_info.thresholds(
        sample_3dgray_array, Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    result10: Any = image_info.thresholds(
        sample_3dgray_array,
        Apply_Mask=True,
        Mask=np.zeros(sample_3dgray_array.shape, np.bool),
    )
    assert not isinstance(result1, np.ndarray)
    assert not isinstance(result2, np.ndarray)
    assert not isinstance(result3, np.ndarray)
    assert not isinstance(result4, np.ndarray)
    assert not isinstance(result5, np.ndarray)
    assert not isinstance(result6, np.ndarray)
    assert not isinstance(result7, np.ndarray)
    assert isinstance(result8, np.ndarray)
    assert not isinstance(result9, np.ndarray)
    assert result10 == 0


def test_euler_num(sample_masks, sample_arrays) -> None:

    test_array: np.ndarray = np.zeros((10, 10), np.bool)
    test_array[2:8, 2:8] = 1
    test_array[4:6, 4:6] = 0
    assert isinstance(image_info.euler_num(sample_masks["2dgray"]), int)
    assert isinstance(image_info.euler_num(sample_masks["3dgray"]), int)
    assert isinstance(image_info.euler_num(sample_masks["2drgb"]), int)
    assert isinstance(image_info.euler_num(sample_masks["3drgb"]), int)
    assert isinstance(image_info.euler_num(sample_arrays["2dgray"]), int)
    assert isinstance(image_info.euler_num(sample_arrays["3dgray"]), int)
    assert isinstance(image_info.euler_num(sample_arrays["2drgb"]), int)
    assert isinstance(image_info.euler_num(sample_arrays["3drgb"]), int)
    assert image_info.euler_num(test_array) == 0


def test_blur_effect(sample_arrays) -> None:

    assert isinstance(image_info.blur_effect(sample_arrays["2dgray"], 3), float)
    assert isinstance(image_info.blur_effect(sample_arrays["3dgray"], 3), float)
    assert isinstance(image_info.blur_effect(sample_arrays["2drgb"], 3), float)
    assert isinstance(image_info.blur_effect(sample_arrays["3drgb"], 3), float)


def test_entropy(sample_arrays) -> None:

    assert isinstance(image_info.shannon_entropy(sample_arrays["2dgray"]), float)
    assert isinstance(image_info.shannon_entropy(sample_arrays["2drgb"]), float)
    assert image_info.shannon_entropy(sample_arrays["3dgray"]) == 0
    assert image_info.shannon_entropy(sample_arrays["3drgb"]) == 0


def test_inertia_tensor_eigs(sample_arrays) -> None:

    assert isinstance(image_info.inertia_tensor_eigs(sample_arrays["2dgray"]), dict)
    assert isinstance(image_info.inertia_tensor_eigs(sample_arrays["3dgray"]), dict)
    assert isinstance(image_info.inertia_tensor_eigs(sample_arrays["2drgb"]), dict)
    assert isinstance(image_info.inertia_tensor_eigs(sample_arrays["3drgb"]), dict)
    assert len(image_info.inertia_tensor_eigs(sample_arrays["2dgray"])) == 2
    assert len(image_info.inertia_tensor_eigs(sample_arrays["3dgray"])) == 3
    assert len(image_info.inertia_tensor_eigs(sample_arrays["2drgb"])) == 2
    assert len(image_info.inertia_tensor_eigs(sample_arrays["3drgb"])) == 3
