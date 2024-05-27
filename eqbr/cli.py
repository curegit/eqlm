from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .core import bi_pp, modes
from .img import load_image, save_image, split_alpha, merge_alpha, color_transforms
from .types import fileinput, choice


def main():
    exit_code = 0
    parser = ArgumentParser(prog="eqbr", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Halftone Converter: an image converter to generate halftone images")
    parser.add_argument("input", metavar="FILE", type=fileinput, help="describe input image files (pass '-' to specify stdin)")
    parser.add_argument("output", metavar="FILE", type=fileinput, nargs="?", help="describe input image files (pass '-' to specify stdin)")
    parser.add_argument("-m", "--mode", choices=list(modes.keys()), help="")
    parser.add_argument("-v", "--version", action="version", version="1")
    parser.add_argument("-F", "--resample", type=choice, choices=["nearest", "linear", "lanczos2", "lanczos3", "spline36"], default="linear", help="resampling method for determining dot size")
    args = parser.parse_args()

    f = args.input
    o = args.output
    n = 3  # w
    m = 2  # h

    x = load_image(f)
    bgr, alpha = split_alpha(x)
    mode = modes[args.mode]
    f, g = color_transforms(mode.value.color, transpose=True)
    a = f(bgr)
    print("Input:", a.shape)
    h = bi_pp(a, channel=mode.value.channel, n=(m, n), alpha=alpha)
    y = g(h)
    y = merge_alpha(y, alpha)
    save_image(y, args.output)
