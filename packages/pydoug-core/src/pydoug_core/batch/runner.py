"""
Headless batch runner to apply a workflow to multiple image datasets.
"""

from collections.abc import Iterator
from pathlib import Path

from numpy import ndarray

from pydoug_core.batch._result_save import default_save_result
from pydoug_core.batch.recorder import OperationSpec, Recorder
from pydoug_core.io.imreader import imread, seqread
from pydoug_core.io.imwriter import imwrite
from pydoug_core.registry import get_operation


def run_batch(
    image_paths: list[Path],
    workflow_dir: Path,
    output_dir: Path,
    save_every_step: bool = False,
    save_images: bool = True,
) -> Iterator:
    """
    Generator to iterateively perform batch processing on images.

    Parameters
    ----------
    image_paths: list[Path]
        A list of paths to the images to be processed.
    workflow_dir: Path
        The directory containing the workflow to aply during processing.
    output_dir: Path
        The output directory where the results should be saved.

    Yields
    ------
    index: int
        The current index (image number) being processed.
    total: int
        The total number of images to be processed.
    """

    recorder: Recorder = Recorder()
    recorder.load(Path(workflow_dir))
    output_dir = Path(output_dir)
    total: int = len(image_paths)

    pad: int = max(2, len(str(len(recorder.steps))))
    step_folders: dict[str, str] = {
        step.step_id: f"{idx:0{pad}d}_{get_operation(step.operation_name).name}"
        for idx, step in enumerate(recorder.steps, start=1)
    }

    for step in recorder.steps:
        spec: OperationSpec = get_operation(step.operation_name)
        needs_folder: bool = spec.output_kind != "image" or (
            save_every_step and save_images
        )
        if needs_folder:
            (output_dir / step_folders[step.step_id]).mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(image_paths, start=1):
        name: str = path.stem if path.is_file() else path.name
        yield index, total, f"Processing image {index} of {total}: {name}"

        primary: ndarray | None = seqread(path) if path.is_dir() else imread(path)
        if primary is None:
            yield index, total, f"Skipped (could not read): {name}"
            continue

        last_image_result: ndarray | None = None
        for step, spec, result in recorder.iter_replay(primary):
            folder: Path = output_dir / step_folders[step.step_id]
            if spec.output_kind == "image":
                last_image_result = result
                if save_every_step and save_images:
                    out_path: Path = folder / f"{name}.tiff"
                    imwrite(result, folder, name, "tiff")
                    if not out_path.exists():
                        text: str = f"WARNING: failed to write {folder.name} for {name}"
                        yield index, total, text
            else:
                (spec.save_result or default_save_result)(result, folder / name)

        if not save_every_step and save_images and last_image_result is not None:
            out_path: Path = output_dir / f"{name}.tiff"
            imwrite(last_image_result, output_dir, name, "tiff")
            if not out_path.exists():
                yield (
                    index,
                    total,
                    f"WARNING: failed to write final image for {name}",
                )

    yield total, total, "Finished processing all images!"
