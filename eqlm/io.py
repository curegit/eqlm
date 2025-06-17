import sys
import os
import io
from io import BufferedIOBase
from pathlib import Path
from numpy import ndarray
from .img import save_image
from .types import AutoUniquePath
from .utils import eprint


def export_png(
    img: ndarray,
    filelike: Path | str | AutoUniquePath | BufferedIOBase | None,
    *,
    icc: bytes | None = None,
    deep: bool = False,
    slow: bool = False,
) -> int:
    exit_code = 0
    if filelike is None:
        try:
            buf = io.BytesIO()
            save_image(img, buf, prefer16=deep, icc_profile=icc, hard=slow)
            sys.stdout.buffer.write(buf.getbuffer())
        except BrokenPipeError:
            exit_code = 128 + 13
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
    elif isinstance(filelike, BufferedIOBase):
        save_image(img, filelike, prefer16=deep, icc_profile=icc, hard=slow)
    else:
        if isinstance(filelike, AutoUniquePath):
            fp, output_path = filelike.open_png()
        else:
            fp = output_path = filelike
        save_image(img, fp, prefer16=deep, icc_profile=icc, hard=slow)
        if Path(output_path).suffix.lower() != os.extsep + "png":
            eprint(f"Warning: The output file extension is not {os.extsep}png")
    return exit_code
