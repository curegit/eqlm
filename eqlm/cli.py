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
        def name(input_path: Path | str, *, suffix: str = "-eqlm", ext=f"{os.extsep}png") -> Path:
            path = Path(input_path).resolve()
            return alt_filepath((Path(".") / (path.stem + suffix)).with_suffix(ext))

    class Average:
        def __str__(self) -> str:
            return "Average"

    exit_code = 0

    try:
        parser = ArgumentParser(prog="eqlm", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Simple CLI tool to spatially equalize image luminance")
        parser.add_argument("input", metavar="IN_FILE", type=fileinput, help="input image file path (use '-' for stdin)")
        parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=Auto(), help="output PNG image file path (use '-' for stdout)")
        parser.add_argument("-v", "--version", action="version", version=version)
        parser.add_argument("-m", "--mode", type=choice, choices=list(modes.keys()), default=list(modes.keys())[0], help="processing mode")
        parser.add_argument("-n", "--divide", metavar=("M", "N"), type=uint, nargs=2, default=(2, 2), help="divide image into MxN blocks for aggregation (note that it doesn't respect Exif orientation)")
        parser.add_argument("-t", "--target", metavar="RATE", type=rate, default=Average(), help="output level target rate, 0.0 (min) to 1.0 (max)")
        parser.add_argument("-c", "--clamp", action="store_true", help="")
        parser.add_argument("-e", "--median", action="store_true", help="aggregate each block using median")
        parser.add_argument("-u", "--unweighted", action="store_true", help="disable alpha channel weighting")
        parser.add_argument("-g", "--gamma", metavar="GAMMA", type=positive, nargs="?", const=2.2, help="apply inverse gamma correction before process [GAMMA=2.2]")
        parser.add_argument("-d", "--depth", type=int, choices=[8, 16], default=8, help="bit depth of the output PNG image")
        args = parser.parse_args()

        input_file: Path | None = args.input
        output_file: Path | Auto | None = args.output
        mode: Mode = modes[args.mode]
        vertical: int | None = args.divide[1] or None
        horizontal: int | None = args.divide[0] or None
        target: float | None = None if isinstance(args.target, Average) else args.target
        clamp: bool = args.clamp
        median: bool = args.median
        unweighted: bool = args.unweighted
        gamma: float | None = args.gamma
        deep: bool = args.depth == 16

        x, icc = load_image(io.BytesIO(sys.stdin.buffer.read()).getbuffer() if input_file is None else input_file, normalize=True)

        eprint(f"Size: {x.shape[1]}x{x.shape[0]}")
        eprint(f"Grid: {horizontal or 1}x{vertical or 1}")
        eprint("Process ...")

        bgr, alpha = split_alpha(x)
        f, g = color_transforms(mode.value.color, gamma=gamma, transpose=True)
        a = f(bgr)
        c = mode.value.channel
        a[c] = biprocess(a[c], n=(vertical, horizontal), alpha=(None if unweighted else alpha), target=target, median=median, clamp=clamp, clip=(mode.value.min, mode.value.max))
        y = merge_alpha(g(a), alpha)

        eprint("Saving ...")

        if output_file is None:
            try:
                buf = io.BytesIO()
                save_image(y, buf, prefer16=deep, icc_profile=icc)
                sys.stdout.buffer.write(buf.getbuffer())
            except BrokenPipeError:
                exit_code = 128 + 13
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
        else:
            if isinstance(output_file, Auto):
                output_path = Auto.name("stdin" if input_file is None else input_file)
            else:
                output_path = output_file
            save_image(y, output_path, prefer16=deep, icc_profile=icc)
            if output_path.suffix.lower() != os.extsep + "png":
                eprint(f"Warning: The output file extension is not {os.extsep}png")
        return exit_code

    except KeyboardInterrupt:
        eprint("KeyboardInterrupt")
        exit_code = 128 + 2
        return exit_code
