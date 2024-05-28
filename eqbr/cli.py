from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .core import biprocess, modes
from .img import load_image, save_image, split_alpha, merge_alpha, color_transforms
from .types import fileinput, choice, uint, positive


def main():
    from . import __version__ as version

    exit_code = 0
    parser = ArgumentParser(prog="eqbr", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Halftone Converter: an image converter to generate halftone images")
    parser.add_argument("input", metavar="FILE", type=fileinput, help="describe input image files (pass '-' to specify stdin)")
    parser.add_argument("output", metavar="FILE", type=fileinput, nargs="?", help="describe input image files (pass '-' to specify stdin)")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-m", "--mode", type=choice, choices=list(modes.keys()), default=list(modes.keys())[0], help="a")
    parser.add_argument("-n", "--divide", metavar=("M", "N"), type=uint, nargs=2, default=(2, 2), help="a")
    parser.add_argument("-g", "--gamma", type=positive, help="gamma correction value")
    args = parser.parse_args()

    f = args.input
    o = args.output
    vertical: int | None = args.divide[1] or None
    horizontal: int | None = args.divide[0] or None
    gamma: float | None = args.gamma

    x = load_image(f)
    bgr, alpha = split_alpha(x)
    mode = modes[args.mode]
    f, g = color_transforms(mode.value.color, gamma=gamma, transpose=True)
    a = f(bgr)
    print("Input:", a.shape)
    c = mode.value.channel
    a[c] = biprocess(a[c], n=(vertical, horizontal), alpha=alpha, clip=(mode.value.min, mode.value.max))

    y = g(a)
    y = merge_alpha(y, alpha)
    save_image(y, o)
