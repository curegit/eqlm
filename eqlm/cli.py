from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from .types import fileinput, fileoutput, choice, uint, positive, ufloat, rate, AutoUniquePath
from .utils import eprint
from .eq.main import equalize
from .eq.core import modes as eq_modes, interpolations
from .match.main import match
from .match.core import modes as match_modes
from .laps.main import laps
from .laps.core import NamedStencil, stencils, modes as laps_modes
from .desc.main import desc


def main(argv: list[str] | None = None) -> int:
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

    try:
        parser = ArgumentParser(prog="eqlm", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Simple CLI tool to manipulate images in various ways")
        parser.add_argument("-v", "--version", action="version", version=version)
        subparsers = parser.add_subparsers(dest="command", required=True, help="command to perform")

        def create_subparser(cmd: str, **kwargs):
            return subparsers.add_parser(cmd, allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, epilog="A '--' is usable to terminate option parsing so remaining arguments are treated as positional arguments.", **kwargs)

        # Original eq command
        eq_parser = create_subparser(eq_sub := "eq", aliases=["equalize"], description="Equalize image lightness, saturation, or brightness", help="equalize image lightness, saturation, or brightness")
        eq_parser.add_argument("input", metavar="IN_FILE", type=fileinput, help="input image file path (use '-' for stdin, '_' for clipboard)")
        eq_parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=AutoUniquePath(), help="output PNG image file path (use '-' for stdout, '_' for clipboard)")
        eq_parser.add_argument("-m", "--mode", type=choice, choices=list(eq_modes.keys()), default=list(eq_modes.keys())[0], help=f"processing mode ({", ".join(f'{k}: {v}' for k, v in eq_modes.items())})")
        eq_parser.add_argument("-n", "--divide", metavar=("M", "N"), type=uint, nargs=2, default=(2, 2), help="divide image into MxN (Horizontal x Vertical) blocks for aggregation")
        eq_parser.add_argument("-i", "--interpolation", type=choice, choices=list(interpolations.keys()), default=list(interpolations.keys())[0], help=f"interpolation method ({", ".join(f"{k}: {v.value}" for k, v in interpolations.items())})")
        eq_parser.add_argument("-t", "--target", metavar="RATE", type=rate, default=Average(), help="set the target rate for the output level, ranging from 0.0 (minimum) to 1.0 (maximum)")
        eq_parser.add_argument("-c", "--clamp", action="store_true", help="clamp the level values in extrapolated boundaries")
        eq_parser.add_argument("-e", "--median", action="store_true", help="aggregate each block using median instead of mean")
        eq_parser.add_argument("-u", "--unweighted", action="store_true", help="disable weighting based on the alpha channel")

        # Match command
        match_parser = create_subparser(match_sub := "match", description="Match histogram of source image to reference image", help="match histogram of source image to reference image")
        match_parser.add_argument("source", metavar="SOURCE_FILE", type=fileinput, help="source image file path (use '-' for stdin, '_' for clipboard)")
        match_parser.add_argument("reference", metavar="REFERENCE_FILE", type=fileinput, help="reference image file path (use '-' for stdin, '_' for clipboard)")
        match_parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=AutoUniquePath(), help="output PNG image file path (use '-' for stdout, '_' for clipboard)")
        match_parser.add_argument("-m", "--mode", type=choice, choices=list(match_modes.keys()), default=list(match_modes.keys())[0], help=f"processing mode ({", ".join(f'{k}: {v}' for k, v in match_modes.items())})")
        group = match_parser.add_mutually_exclusive_group()
        group.add_argument("-a", "--alpha", metavar=("SOURCE", "REFERENCE"), type=rate, nargs=2, default=(0.0, 0.5), help="cutout threshold for the alpha channel (source, reference)")
        group.add_argument("-u", "--unweighted", action="store_true", help="disable cutout based on the alpha channel")

        # Laps command
        laps_parser = create_subparser(laps_sub := "laps", description="Sharpen an image using a Laplacian variant kernel", help="sharpen an image using a Laplacian variant kernel")
        laps_parser.add_argument("input", metavar="IN_FILE", type=fileinput, help="input image file path (use '-' for stdin, '_' for clipboard)")
        laps_parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=AutoUniquePath(), help="output PNG image file path (use '-' for stdout, '_' for clipboard)")
        laps_parser.add_argument("-m", "--mode", type=choice, choices=list(laps_modes.keys()), default=list(laps_modes.keys())[0], help=f"processing channel mode ({", ".join(f'{k}: {v}' for k, v in laps_modes.items())})")
        laps_parser.add_argument("-t", "--stencil", type=choice, choices=list(stencils.keys()), default=next(k for k, v in stencils.items() if v == NamedStencil.OonoPuri), help=f"kernel selection ({", ".join(f'{k}: {v.description}' for k, v in stencils.items())})")
        laps_parser.add_argument("-c", "--coef", metavar="C", type=ufloat, default=0.2, help="sharpening factor")
        laps_parser.add_argument("-a", "--include-alpha", action="store_true", help="also sharpen the alpha channel")

        # Desc command
        desc_parser = create_subparser(desc_sub := "desc", description="Fourier Transform-based descreening for scanned images", help="Fourier Transform-based descreening for scanned images")
        desc_parser.add_argument("input", metavar="IN_FILE", type=fileinput, help="input image file path (use '-' for stdin, '_' for clipboard)")
        desc_parser.add_argument("output", metavar="OUT_FILE", type=fileoutput, nargs="?", default=AutoUniquePath(), help="output PNG image file path (use '-' for stdout, '_' for clipboard)")
        desc_parser.add_argument("--cmyk", action=BooleanOptionalAction, default=False, help=f"switch to perform descreening in CMYK color space")
        desc_parser.add_argument("--nl-means", action=BooleanOptionalAction, default=True, help=f"switch to apply Non-Local Means denoising after descreening")

        # Shared arguments
        ParserStack(eq_parser, match_parser, laps_parser, desc_parser).add_argument("-g", "--gamma", metavar="GAMMA", type=positive, nargs="?", const=2.2, help="apply inverse gamma correction before the process [GAMMA=2.2]")
        ParserStack(eq_parser, match_parser, laps_parser).add_argument("-d", "--depth", type=int, choices=[8, 16], default=8, help="bit depth of the output PNG image")
        ParserStack(eq_parser, match_parser, laps_parser, desc_parser).add_argument("-s", "--slow", action="store_true", help="use the highest PNG compression level")
        ParserStack(eq_parser, match_parser, laps_parser, desc_parser).add_argument("-x", "--no-orientation", dest="no_orientation", action="store_true", help="ignore the Exif orientation metadata")

        args = parser.parse_args(argv)
        match args.command:
            case str() as command if command == eq_sub:
                return equalize(
                    input_file=args.input,
                    output_file=args.output,
                    mode=eq_modes[args.mode],
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
            case str() as command if command == match_sub:
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
            case str() as command if command == laps_sub:
                return laps(
                    input_file=args.input,
                    output_file=args.output,
                    mode=laps_modes[args.mode],
                    stencil=stencils[args.stencil],
                    coef=args.coef,
                    include_alpha=args.include_alpha,
                    gamma=args.gamma,
                    deep=(args.depth == 16),
                    slow=args.slow,
                    orientation=(not args.no_orientation),
                )
            case str() as command if command == desc_sub:
                return desc(
                    input_file=args.input,
                    output_file=args.output,
                    cmyk=args.cmyk,
                    nl_means=args.nl_means,
                    gamma=args.gamma,
                    slow=args.slow,
                    orientation=(not args.no_orientation),
                )
            case _:
                raise AssertionError()

    except KeyboardInterrupt:
        eprint("KeyboardInterrupt")
        exit_code = 128 + 2
        return exit_code
