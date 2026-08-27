"""
Tests for napari plugin _writer module.
"""

from pydoug_napari import _writer


def test_write_image_returns(tmp_path, sample_3dgray_array) -> None:

    result1: list[str] | None = _writer.write_image(
        tmp_path / "test_write.tiff", sample_3dgray_array, {}
    )
    assert isinstance(result1, list)
