from io import BufferedIOBase
from pathlib import Path
from .core import Mode, histogram_matching
from ..img import split_alpha, merge_alpha, color_transforms
from ..io import import_image, export_png
from ..types import AutoUniquePath
from ..utils import eprint


def match(*, source_file: Path | str | bytes | None, reference_file: Path | str | bytes | None, output_file: Path | str | AutoUniquePath | BufferedIOBase | None, mode: Mode, alpha: tuple[float | None, float | None] = (0.0, 0.5), gamma: float | None = None, deep: bool = False, slow: bool = False, orientation: bool = True) -> int:
    exit_code = 0

    if source_file is None and reference_file is None:
        raise ValueError("Cannot specify reading from stdin for both source and reference images simultaneously")
    x, icc = import_image(source_file, normalize=True, orientation=orientation)
    r, ref_icc = import_image(reference_file, normalize=True, orientation=orientation)

    eprint("Process ...")

    bgr, alpha_x = split_alpha(x)
    bgr_ref, alpha_ref = split_alpha(r)
    f, g = color_transforms(mode.value.color, gamma=gamma, transpose=True)
    a = f(bgr)
    b = f(bgr_ref)
    c = mode.value.channels
    alpha_cutout = None if alpha[0] is None else alpha_x
    alpha_cutout_ref = None if alpha[1] is None else alpha_ref
    matched = histogram_matching(a, b, c, x_alpha=alpha_cutout, r_alpha=alpha_cutout_ref, x_alpha_threshold=(alpha[0] or 0.0), r_alpha_threshold=(alpha[1] or 0.0))
    y = merge_alpha(g(matched), alpha_x)

    eprint("Saving ...")

    if isinstance(output_file, AutoUniquePath):
        output_file.input_path = "stdin" if source_file is None else "memory" if isinstance(source_file, bytes) else source_file
        output_file.suffix = f"-matched-{mode.name.lower()}"
    if (return_code := export_png(y, output_file, deep=deep, slow=slow, icc=ref_icc)) != 0:
        exit_code = return_code

    return exit_code
