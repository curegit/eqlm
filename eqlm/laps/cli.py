from io import BufferedIOBase
from pathlib import Path
from .core import NamedStencil, laplacian_sharpening, Mode
from ..img import split_alpha, merge_alpha, color_transforms
from ..io import import_image, export_png
from ..types import AutoUniquePath
from ..utils import eprint


def laps(*, input_file: Path | str | bytes | None, output_file: Path | str | AutoUniquePath | BufferedIOBase | None, mode: Mode, stencil: NamedStencil, coef: float = 0.2, include_alpha: bool = False, gamma: float | None = None, deep: bool = False, slow: bool = False, orientation: bool = True) -> int:
    exit_code = 0

    x, icc = import_image(input_file, normalize=True, orientation=orientation)

    eprint("Process ...")

    bgr, alpha = split_alpha(x)
    f, g = color_transforms(mode.value.color, gamma=gamma, transpose=False)
    a = f(bgr)
    b = laplacian_sharpening(
        a,
        stencil.value,
        channels=mode.value.channels,
        coef=coef,
        clip=list(
            zip(
                [mi if isinstance(mi := mode.value.min, float) else mi[i] for i in range(len(mode.value.channels))],
                [ma if isinstance(ma := mode.value.max, float) else ma[i] for i in range(len(mode.value.channels))],
            )
        ),
    )
    if alpha is not None and include_alpha:
        alpha = laplacian_sharpening(alpha.reshape((*alpha.shape, 1)), stencil.value, channels=[0], coef=coef, clip=(0.0, 1.0)).reshape(alpha.shape)
    y = merge_alpha(g(b), alpha)

    eprint("Saving ...")

    if isinstance(output_file, AutoUniquePath):
        output_file.input_path = "stdin" if input_file is None else "memory" if isinstance(input_file, bytes) else input_file
        output_file.suffix = f"-laps-{stencil.name.lower()}"
    if (return_code := export_png(y, output_file, deep=deep, slow=slow, icc=icc)) != 0:
        exit_code = return_code

    return exit_code
