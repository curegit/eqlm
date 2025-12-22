import numpy as np
import cv2
from io import BufferedIOBase
from pathlib import Path
from .core import descreen
from ..img import split_alpha, merge_alpha, color_transforms, Color
from ..io import import_image, export_png
from ..types import AutoUniquePath
from ..utils import eprint


def desc(*, input_file: Path | str | bytes | None, output_file: Path | str | AutoUniquePath | BufferedIOBase | None, cmyk: bool = False, nl_means: bool = True, gamma: float | None = None, slow: bool = False, orientation: bool = True):
    exit_code = 0

    x, icc = import_image(input_file, normalize=True, orientation=orientation)

    eprint("Process ...")

    bgr, alpha = split_alpha(x)
    f, g = color_transforms(Color.RGB, gamma=gamma, transpose=True)
    a = f(bgr)
    b = descreen(a, cmyk=cmyk)

    z = g(b)
    if nl_means:
        eprint("Denoise ...")
        # Type of input image must be CV_8UC3 or CV_8UC4 for now
        z = cv2.fastNlMeansDenoisingColored((z * 255).astype(np.uint8), h=6, hColor=6, templateWindowSize=5, searchWindowSize=9)
        z = (z / 255).astype(np.float32)
    y = merge_alpha(z, alpha)

    eprint("Saving ...")

    if isinstance(output_file, AutoUniquePath):
        output_file.input_path = "stdin" if input_file is None else "memory" if isinstance(input_file, bytes) else input_file
        output_file.suffix = "-desc"
    if (return_code := export_png(y, output_file, deep=False, slow=slow, icc=icc)) != 0:
        exit_code = return_code

    return exit_code
