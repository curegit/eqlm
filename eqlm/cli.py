import sys
import os
import io
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .core import Mode, Interpolation, biprocess, modes, interpolations
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
        def open_named(input_path: Path | str, *, suffix: str = "-eqlm", ext=f"{os.extsep}png"):
            path = Path(input_path).resolve()
            filepath = (Path(".") / (path.stem + suffix)).with_suffix(ext)
            while True:
                try:
                    return open(filepath, "xb"), filepath
                except FileExistsError:
                    filepath = alt_filepath(filepath)

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
        parser.add_argument("-n", "--divide", metavar=("M", "N"), type=uint, nargs=2, default=(2, 2), help="divide image into MxN (Horizontal x Vertical) blocks for aggregation")
        parser.add_argument("-i", "--interpolation", type=choice, choices=list(interpolations.keys()), default=list(interpolations.keys())[0], help=f"interpolation method ({", ".join(f"{k}: {v.value}" for k, v in interpolations.items())})")
        parser.add_argument("-t", "--target", metavar="RATE", type=rate, default=Average(), help="set the target rate for the output level, ranging from 0.0 (minimum) to 1.0 (maximum)")
        parser.add_argument("-c", "--clamp", action="store_true", help="clamp the level values in extrapolated boundaries")
        parser.add_argument("-e", "--median", action="store_true", help="aggregate each block using median instead of mean")
        parser.add_argument("-u", "--unweighted", action="store_true", help="disable weighting based on the alpha channel")
        parser.add_argument("-g", "--gamma", metavar="GAMMA", type=positive, nargs="?", const=2.2, help="apply inverse gamma correction before process [GAMMA=2.2]")
        parser.add_argument("-d", "--depth", type=int, choices=[8, 16], default=8, help="bit depth of the output PNG image")
        parser.add_argument("-s", "--slow", action="store_true", help="use the highest PNG compression level")
        parser.add_argument("-x", "--no-orientation", dest="no_orientation", action="store_true", help="ignore the Exif orientation metadata")
        args = parser.parse_args()

        input_file: Path | None = args.input
        output_file: Path | Auto | None = args.output
        mode: Mode = modes[args.mode]
        vertical: int | None = args.divide[1] or None
        horizontal: int | None = args.divide[0] or None
        interpolation: Interpolation = interpolations[args.interpolation]
        target: float | None = None if isinstance(args.target, Average) else args.target
        clamp: bool = args.clamp
        median: bool = args.median
        unweighted: bool = args.unweighted
        gamma: float | None = args.gamma
        deep: bool = args.depth == 16
        slow: bool = args.slow
        orientation: bool = not args.no_orientation

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

    except KeyboardInterrupt:
        eprint("KeyboardInterrupt")
        exit_code = 128 + 2
        return exit_code
