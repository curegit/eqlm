import sys
import io
from io import BufferedIOBase
from pathlib import Path
from .core import Mode, Interpolation, biprocess
from ..img import load_image, split_alpha, merge_alpha, color_transforms
from ..io import export_png
from ..types import AutoUniquePath
from ..utils import eprint


def equalize(*, input_file: Path | str | bytes | None, output_file: Path | str | AutoUniquePath | BufferedIOBase | None, mode: Mode, vertical: int | None, horizontal: int | None, interpolation: Interpolation, target: float | None, clamp: bool, median: bool, unweighted: bool, gamma: float | None, deep: bool, slow: bool, orientation: bool) -> int:
    exit_code = 0

    x, icc = load_image(io.BytesIO(sys.stdin.buffer.read()).getbuffer() if input_file is None else input_file, normalize=True, orientation=orientation)

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
    if (special_code := export_png(y, output_file, deep=deep, slow=slow, icc=icc)) != 0:
        exit_code = special_code

    return exit_code
