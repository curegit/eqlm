from io import BufferedIOBase
from pathlib import Path
from .core import Mode, Interpolation, biprocess
from ..img import split_alpha, merge_alpha, color_transforms
from ..io import import_image, export_png
from ..types import AutoUniquePath
from ..utils import eprint


def equalize(*, input_file: Path | str | bytes | None, output_file: Path | str | AutoUniquePath | BufferedIOBase | None, mode: Mode, vertical: int | None = None, horizontal: int | None = None, interpolation: Interpolation = Interpolation.Linear, target: float | None = None, clamp: bool = False, median: bool = False, unweighted: bool = False, gamma: float | None = None, deep: bool = False, slow: bool = False, orientation: bool = True) -> int:
    exit_code = 0

    x, icc = import_image(input_file, normalize=True, orientation=orientation)

    eprint(f"Size: {x.shape[1]}x{x.shape[0]}")
    eprint(f"Grid: {horizontal or 1}x{vertical or 1}")
    eprint("Process ...")

    bgr, alpha = split_alpha(x)
    f, g = color_transforms(mode.value.color, gamma=gamma, transpose=True)
    a = f(bgr)
    c = mode.value.channel
    a[c] = biprocess(a[c], n=(vertical, horizontal), alpha=(None if unweighted else alpha), interpolation=(interpolation, interpolation), target=target, median=median, clamp=clamp, clip=(mode.value.min, mode.value.max))
    y = merge_alpha(g(a), alpha)

    eprint("Saving ...")

    if isinstance(output_file, AutoUniquePath):
        output_file.input_path = "stdin" if input_file is None else "memory" if isinstance(input_file, bytes) else input_file
        output_file.suffix = f"-eq-{mode.name.lower()}"
    if (return_code := export_png(y, output_file, deep=deep, slow=slow, icc=icc)) != 0:
        exit_code = return_code

    return exit_code
