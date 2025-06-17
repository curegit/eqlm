import sys
import os
import io
from pathlib import Path
from .core import Mode, Interpolation, biprocess
from ..img import load_image, save_image, split_alpha, merge_alpha, color_transforms
from ..types import Auto
from ..utils import eprint


def equalize(*, input_file: Path | str | None, output_file: Path | str | Auto | None, mode: Mode, vertical: int | None, horizontal: int | None, interpolation: Interpolation, target: float | None, clamp: bool, median: bool, unweighted: bool, gamma: float | None, deep: bool, slow: bool, orientation: bool) -> int:
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

    if output_file is None:
        try:
            buf = io.BytesIO()
            save_image(y, buf, prefer16=deep, icc_profile=icc, hard=slow)
            sys.stdout.buffer.write(buf.getbuffer())
        except BrokenPipeError:
            exit_code = 128 + 13
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
    else:
        if isinstance(output_file, Auto):
            fp, output_path = Auto.open_named("stdin" if input_file is None else input_file)
        else:
            fp = output_path = output_file
        save_image(y, fp, prefer16=deep, icc_profile=icc, hard=slow)
        if output_path.suffix.lower() != os.extsep + "png":
            eprint(f"Warning: The output file extension is not {os.extsep}png")
    return exit_code
