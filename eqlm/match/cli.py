import sys
import os
import io
from pathlib import Path
from .core import Mode, histgram_matching
from ..img import load_image, save_image, split_alpha, merge_alpha, color_transforms
from ..types import Auto
from ..utils import eprint


# TODO: icc matching to reference
def match(*, source_file: Path | None, reference_file: Path | None, output_file: Path | Auto | None, mode: Mode, alpha: tuple[float | None, float | None] = (0.0, 0.5), gamma: float | None, deep: bool, slow: bool, orientation: bool):
    exit_code = 0

    x, icc = load_image(io.BytesIO(sys.stdin.buffer.read()).getbuffer() if source_file is None else source_file, normalize=True, orientation=orientation)
    r, ref_icc = load_image(io.BytesIO(sys.stdin.buffer.read()).getbuffer() if reference_file is None else reference_file, normalize=True, orientation=orientation)

    eprint("Process ...")

    bgr, alpha_x = split_alpha(x)
    bgr_ref, alpha_ref = split_alpha(r)
    f, g = color_transforms(mode.value.color, gamma=gamma, transpose=True)
    a = f(bgr)
    b = f(bgr_ref)
    c = mode.value.channels

    alpha_cutout = None if alpha[0] is None else alpha_x
    alpha_cutout_ref = None if alpha[1] is None else alpha_ref
    matched = histgram_matching(a, b, c, x_alpha=alpha_cutout, r_alpha=alpha_cutout_ref, x_alpha_threshold=(alpha[0] or 0.0), r_alpha_threshold=(alpha[1] or 0.0))

    y = merge_alpha(g(matched), alpha_x)

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
            fp, output_path = Auto.open_named("stdin" if source_file is None else source_file, suffix=f"-matched-{mode.name.lower()}")
        else:
            fp = output_path = output_file
        save_image(y, fp, prefer16=deep, icc_profile=icc, hard=slow)
        if output_path.suffix.lower() != os.extsep + "png":
            eprint(f"Warning: The output file extension is not {os.extsep}png")
    return exit_code
