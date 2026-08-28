"""
Tests for napari plugin _writer module.
"""

from pydoug_napari import _writer


def test_write_image_returns(tmp_path, sample_3dgray_array) -> None:

    result: list[str] | None = _writer.write_image(
        tmp_path / "test_write.tiff", sample_3dgray_array, {}
    )
    assert isinstance(result, list)


def test_write_image_sequence_returns(tmp_path, sample_3dgray_array) -> None:

    result: list[str] | None = _writer.write_image_sequence(
        tmp_path / "test_write_seq", sample_3dgray_array, {}
    )
    assert isinstance(result, list)
