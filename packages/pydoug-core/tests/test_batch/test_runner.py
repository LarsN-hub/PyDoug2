"""
Tests for PyDoug's batch.runner module.
"""

import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pydoug_core.batch import recorder, runner


class DummyLayer:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data = data


def test_run_batch(
    tmp_path,
    sample_2dgray_array,
    registered_nothing,
    registered_nothing_plot,
    registered_nothing_calc,
) -> None:

    os.mkdir(Path(tmp_path / "test_runner_import"))
    os.mkdir(Path(tmp_path / "test_runner_output1"))
    os.mkdir(Path(tmp_path / "test_runner_output2"))
    os.mkdir(Path(tmp_path / "test_runner_output3"))
    assert Path(tmp_path / "test_runner_import").exists()
    assert Path(tmp_path / "test_runner_output1").exists()
    assert Path(tmp_path / "test_runner_output2").exists()
    assert Path(tmp_path / "test_runner_output3").exists()
    iio.imwrite(Path(tmp_path / "test_runner_import/test1.tiff"), sample_2dgray_array)
    iio.imwrite(Path(tmp_path / "test_runner_import/test2.tiff"), sample_2dgray_array)
    assert Path(tmp_path / "test_runner_import/test1.tiff").exists()
    assert Path(tmp_path / "test_runner_import/test2.tiff").exists()
    layer1: DummyLayer = DummyLayer(sample_2dgray_array)
    record: recorder.Recorder = recorder.Recorder()
    record.record("_nothing", {}, {"layer": "primary"}, layer1)
    record.record(
        "_nothing_plot", {}, {"layer": f"step:{record.step_id_for_layer(layer1)}"}
    )
    record.record(
        "_nothing_calc", {}, {"layer": f"step:{record.step_id_for_layer(layer1)}"}
    )
    assert len(record.steps) == 3
    record.save(Path(tmp_path / "test_runner_workflow"))
    assert Path(tmp_path / "test_runner_workflow").exists()

    result1: list[tuple[int, int, str]] = list(
        runner.run_batch(
            [
                Path(tmp_path / "test_runner_import/test1.tiff"),
                Path(tmp_path / "test_runner_import/test2.tiff"),
            ],
            Path(tmp_path / "test_runner_workflow"),
            Path(tmp_path / "test_runner_output1"),
        )
    )
    assert len(result1) == 3
    assert result1[0] == (1, 2, "Processing image 1 of 2: test1")
    assert result1[1] == (2, 2, "Processing image 2 of 2: test2")
    assert result1[2] == (2, 2, "Finished processing all images!")
    assert Path(tmp_path / "test_runner_output1/test1.tiff").exists()
    assert Path(tmp_path / "test_runner_output1/test2.tiff").exists()
    assert Path(tmp_path / "test_runner_output1/02__nothing_plot/test1.png").exists()
    assert Path(tmp_path / "test_runner_output1/02__nothing_plot/test2.png").exists()
    assert Path(tmp_path / "test_runner_output1/03__nothing_calc/test1.json").exists()
    assert Path(tmp_path / "test_runner_output1/03__nothing_calc/test2.json").exists()

    result2: list[tuple[int, int, str]] = list(
        runner.run_batch(
            [
                Path(tmp_path / "test_runner_import/test1.tiff"),
                Path(tmp_path / "test_runner_import/test2.tiff"),
            ],
            Path(tmp_path / "test_runner_workflow"),
            Path(tmp_path / "test_runner_output2"),
            save_every_step=True,
        )
    )
    assert result1 == result2
    assert Path(tmp_path / "test_runner_output2/01__nothing/test1.tiff").exists()
    assert Path(tmp_path / "test_runner_output2/01__nothing/test2.tiff").exists()
    assert Path(tmp_path / "test_runner_output2/02__nothing_plot/test1.png").exists()
    assert Path(tmp_path / "test_runner_output2/02__nothing_plot/test2.png").exists()
    assert Path(tmp_path / "test_runner_output2/03__nothing_calc/test1.json").exists()
    assert Path(tmp_path / "test_runner_output2/03__nothing_calc/test2.json").exists()

    result3: list[tuple[int, int, str]] = list(
        runner.run_batch(
            [
                Path(tmp_path / "test_runner_import/test1.tiff"),
                Path(tmp_path / "test_runner_import/test2.tiff"),
            ],
            Path(tmp_path / "test_runner_workflow"),
            Path(tmp_path / "test_runner_output3"),
            save_images=False,
        )
    )
    assert result1 == result3
    assert len(os.listdir(Path(tmp_path / "test_runner_output3"))) == 2
    assert Path(tmp_path / "test_runner_output3/02__nothing_plot/test1.png").exists()
    assert Path(tmp_path / "test_runner_output3/02__nothing_plot/test2.png").exists()
    assert Path(tmp_path / "test_runner_output3/03__nothing_calc/test1.json").exists()
    assert Path(tmp_path / "test_runner_output3/03__nothing_calc/test2.json").exists()


def test_run_batch_no_primary(
    tmp_path,
    sample_2dgray_array,
    registered_nothing,
) -> None:

    os.mkdir(Path(tmp_path / "test_runner_output_no_primary"))
    assert Path(tmp_path / "test_runner_output_no_primary").exists()
    layer1: DummyLayer = DummyLayer(sample_2dgray_array)
    record: recorder.Recorder = recorder.Recorder()
    record.record("_nothing", {}, {"layer": "primary"}, layer1)
    record.save(Path(tmp_path / "test_runner_workflow_no_primary"))
    assert Path(tmp_path / "test_runner_workflow_no_primary").exists()

    result: list[tuple[int, int, str]] = list(
        runner.run_batch(
            [
                Path(tmp_path / "test_runner_import_no_primary/test1.tiff"),
                Path(tmp_path / "test_runner_import_no_primary/test2.tiff"),
            ],
            Path(tmp_path / "test_runner_workflow_no_primary"),
            Path(tmp_path / "test_runner_output_no_primary"),
        )
    )
    assert len(result) == 5
    assert result[0] == (1, 2, "Processing image 1 of 2: test1.tiff")
    assert result[1] == (1, 2, "Skipped (could not read): test1.tiff")
    assert result[2] == (2, 2, "Processing image 2 of 2: test2.tiff")
    assert result[3] == (2, 2, "Skipped (could not read): test2.tiff")
    assert result[4] == (2, 2, "Finished processing all images!")


def test_run_batch_no_write(
    tmp_path, sample_2dgray_array, registered_nothing, monkeypatch
) -> None:

    os.mkdir(Path(tmp_path / "test_runner_import_no_write"))
    os.mkdir(Path(tmp_path / "test_runner_output_no_write1"))
    os.mkdir(Path(tmp_path / "test_runner_output_no_write2"))
    assert Path(tmp_path / "test_runner_import_no_write").exists()
    assert Path(tmp_path / "test_runner_output_no_write1").exists()
    assert Path(tmp_path / "test_runner_output_no_write2").exists()
    iio.imwrite(
        Path(tmp_path / "test_runner_import_no_write/test1.tiff"), sample_2dgray_array
    )
    iio.imwrite(
        Path(tmp_path / "test_runner_import_no_write/test2.tiff"), sample_2dgray_array
    )
    layer1: DummyLayer = DummyLayer(sample_2dgray_array)
    record: recorder.Recorder = recorder.Recorder()
    record.record("_nothing", {}, {"layer": "primary"}, layer1)
    record.save(Path(tmp_path / "test_runner_workflow_no_write"))
    assert Path(tmp_path / "test_runner_workflow_no_write").exists()

    def _skip(*args, **kwargs) -> None:
        return

    monkeypatch.setattr(runner, "imwrite", _skip)
    result1: list[tuple[int, int, str]] = list(
        runner.run_batch(
            [
                Path(tmp_path / "test_runner_import_no_write/test1.tiff"),
                Path(tmp_path / "test_runner_import_no_write/test2.tiff"),
            ],
            Path(tmp_path / "test_runner_workflow_no_write"),
            Path(tmp_path / "test_runner_output_no_write1"),
        )
    )
    assert len(result1) == 5
    assert result1[0] == (1, 2, "Processing image 1 of 2: test1")
    assert result1[1] == (1, 2, "WARNING: failed to write final image for test1")
    assert result1[2] == (2, 2, "Processing image 2 of 2: test2")
    assert result1[3] == (2, 2, "WARNING: failed to write final image for test2")
    assert result1[4] == (2, 2, "Finished processing all images!")

    result2: list[tuple[int, int, str]] = list(
        runner.run_batch(
            [
                Path(tmp_path / "test_runner_import_no_write/test1.tiff"),
                Path(tmp_path / "test_runner_import_no_write/test2.tiff"),
            ],
            Path(tmp_path / "test_runner_workflow_no_write"),
            Path(tmp_path / "test_runner_output_no_write2"),
            True,
        )
    )
    assert len(result2) == 5
    assert result2[0] == (1, 2, "Processing image 1 of 2: test1")
    assert result2[1] == (1, 2, "WARNING: failed to write 01__nothing for test1")
    assert result2[2] == (2, 2, "Processing image 2 of 2: test2")
    assert result2[3] == (2, 2, "WARNING: failed to write 01__nothing for test2")
    assert result2[4] == (2, 2, "Finished processing all images!")
