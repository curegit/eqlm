import sys
import os
import io
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .core import Mode, biprocess, modes
from .img import load_image, save_image, split_alpha, merge_alpha, color_transforms
from .types import fileinput, fileoutput, choice, uint, positive, rate
from .utils import alt_filepath

def main() -> int:
    from . import __version__ as version

    def eprint(*args, **kwargs):
        print(*args, **kwargs, file=sys.stderr)

    class Auto:

        def __str__(self) -> str:
            return "Auto"

        @staticmethod
        def name(input_path: Path | str, suffix: str="-eqlm", ext=".png") -> Path:
            p = Path(input_path).resolve()
            return alt_filepath(p.with_stem(p.stem + suffix).with_suffix(ext))

    exit_code = 0
    try:
        parser = ArgumentParser(prog="eqbr", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Halftone Converter: an image converter to generate halftone images")
        parser.add_argument("input", metavar="IN_FILE", type=fileinput, help="describe input image files (pass '-' to specify stdin)")
        parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=Auto(), help="describe input image files (pass '-' to specify stdin)")
        parser.add_argument("-v", "--version", action="version", version=version)
        parser.add_argument("-m", "--mode", type=choice, choices=list(modes.keys()), default=list(modes.keys())[0], help="a")
        parser.add_argument("-n", "--divide", metavar=("M", "N"), type=uint, nargs=2, default=(2, 2), help="a")
        parser.add_argument("-t", "--target", type=rate, help="")
        parser.add_argument("-e", "--median", action="store_true", help="")
        parser.add_argument("-g", "--gamma", type=positive, nargs="?", const=2.2, help="gamma correction value")
        parser.add_argument("-d", "--depth", type=int, choices=[8, 16], default=8, help="a")
        args = parser.parse_args()

        inp: Path | None = args.input
        o: Path | None = Auto.name("stdin" if inp is None else inp) if isinstance(args.output, Auto) else args.output
        mode: Mode = modes[args.mode]
        vertical: int | None = args.divide[1] or None
        horizontal: int | None = args.divide[0] or None
        target: float | None = args.target
        median: bool = args.median
        gamma: float | None = args.gamma
        deep: bool = args.depth == 16



        eprint("Process ...")

        x, icc = load_image(io.BytesIO(sys.stdin.buffer.read()).getbuffer() if inp is None else inp, normalize=True)
        bgr, alpha = split_alpha(x)

        f, g = color_transforms(mode.value.color, gamma=gamma, transpose=True)
        a = f(bgr)
        c = mode.value.channel
        a[c] = biprocess(a[c], n=(vertical, horizontal), alpha=alpha, target=target, median=median, clip=(mode.value.min, mode.value.max))

        y = g(a)
        y = merge_alpha(y, alpha)
        eprint("Saving ...")


        if o is None:
            try:
                with io.BytesIO() as buf:
                    save_image(y, buf, prefer16=deep, icc_profile=icc)
                    sys.stdout.buffer.write(buf.getbuffer())
            except BrokenPipeError:
                exit_code = 128 + 13
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
        else:
                save_image(y, o, prefer16=deep, icc_profile=icc)
        return exit_code

    except KeyboardInterrupt:
        eprint("KeyboardInterrupt")
        exit_code = 128 + 2
        return exit_code
