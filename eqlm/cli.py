from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .types import fileinput, fileoutput, choice, uint, positive, rate, Auto
from .utils import eprint
from .eq.cli import equalize
from .eq.core import modes, interpolations
from .match.cli import match
from .match.core import modes as match_modes


def main() -> int:
    from . import __version__ as version

    class ParserStack:
        def __init__(self, *parsers: ArgumentParser):
            self.parsers = parsers

        def add_argument(self, *args, **kwargs):
            for parser in self.parsers:
                parser.add_argument(*args, **kwargs)

    class Average:
        def __str__(self) -> str:
            return "Average"

    exit_code = 0

    try:
        parser = ArgumentParser(prog="eqlm", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Simple CLI tool to spatially equalize image luminance")
        parser.add_argument("-v", "--version", action="version", version=version)
        subparsers = parser.add_subparsers(dest="command", required=True, help="Commands")

        # Original eq command
        eq_sub = "eq"
        eq_parser = subparsers.add_parser(eq_sub, allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, help="equalize image luminance")
        eq_parser.add_argument("input", metavar="IN_FILE", type=fileinput, help="input image file path (use '-' for stdin)")
        eq_parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=Auto(), help="output PNG image file path (use '-' for stdout)")
        eq_parser.add_argument("-m", "--mode", type=choice, choices=list(modes.keys()), default=list(modes.keys())[0], help="processing mode")
        eq_parser.add_argument("-n", "--divide", metavar=("M", "N"), type=uint, nargs=2, default=(2, 2), help="divide image into MxN (Horizontal x Vertical) blocks for aggregation")
        eq_parser.add_argument("-i", "--interpolation", type=choice, choices=list(interpolations.keys()), default=list(interpolations.keys())[0], help=f"interpolation method ({", ".join(f"{k}: {v.value}" for k, v in interpolations.items())})")
        eq_parser.add_argument("-t", "--target", metavar="RATE", type=rate, default=Average(), help="set the target rate for the output level, ranging from 0.0 (minimum) to 1.0 (maximum)")
        eq_parser.add_argument("-c", "--clamp", action="store_true", help="clamp the level values in extrapolated boundaries")
        eq_parser.add_argument("-e", "--median", action="store_true", help="aggregate each block using median instead of mean")
        eq_parser.add_argument("-u", "--unweighted", action="store_true", help="disable weighting based on the alpha channel")

        # Match command
        match_sub = "match"
        match_parser = subparsers.add_parser(match_sub, allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, help="match histogram of source image to reference image")
        match_parser.add_argument("source", metavar="SOURCE_FILE", type=fileinput, help="source image file path (use '-' for stdin)")
        match_parser.add_argument("reference", metavar="REFERENCE_FILE", type=fileinput, help="reference image file path (use '-' for stdin)")
        match_parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=Auto(), help="output PNG image file path (use '-' for stdout)")
        match_parser.add_argument("-m", "--mode", type=choice, choices=list(match_modes.keys()), default=list(match_modes.keys())[0], help="processing mode")
        match_parser.add_argument("-a", "--alpha", metavar=("SOURCE", "REFERENCE"), type=rate, nargs=2, default=(0.0, 0.5), help="cutout threshold for the alpha channel (source, reference)")
        match_parser.add_argument("-u", "--unweighted", action="store_true", help="disable cutout based on the alpha channel")

        # Shared arguments
        ParserStack(eq_parser, match_parser).add_argument("-g", "--gamma", metavar="GAMMA", type=positive, nargs="?", const=2.2, help="apply inverse gamma correction before process [GAMMA=2.2]")
        ParserStack(eq_parser, match_parser).add_argument("-d", "--depth", type=int, choices=[8, 16], default=8, help="bit depth of the output PNG image")
        ParserStack(eq_parser, match_parser).add_argument("-s", "--slow", action="store_true", help="use the highest PNG compression level")
        ParserStack(eq_parser, match_parser).add_argument("-x", "--no-orientation", dest="no_orientation", action="store_true", help="ignore the Exif orientation metadata")

        args = parser.parse_args()
        match args.command:
            case command if command == eq_sub:
                return equalize(
                    input_file=args.input,
                    output_file=args.output,
                    mode=modes[args.mode],
                    vertical=(args.divide[1] or None),
                    horizontal=(args.divide[0] or None),
                    interpolation=interpolations[args.interpolation],
                    target=(None if isinstance(args.target, Average) else args.target),
                    clamp=args.clamp,
                    median=args.median,
                    unweighted=args.unweighted,
                    gamma=args.gamma,
                    deep=(args.depth == 16),
                    slow=args.slow,
                    orientation=(not args.no_orientation),
                )
            case command if command == match_sub:
                return match(
                    source_file=args.source,
                    reference_file=args.reference,
                    output_file=args.output,
                    mode=match_modes[args.mode],
                    alpha=((None, None) if args.unweighted else args.alpha),
                    gamma=args.gamma,
                    deep=(args.depth == 16),
                    slow=args.slow,
                    orientation=(not args.no_orientation),
                )
            case _:
                raise ValueError()

    except KeyboardInterrupt:
        eprint("KeyboardInterrupt")
        exit_code = 128 + 2
        return exit_code
