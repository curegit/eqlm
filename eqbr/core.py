import sys
import contextlib
from time import time
from glob import glob
from os.path import isfile
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image

# from rich.console import Console
# from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

import numpy as np
from PIL import Image
from numpy import rint, asarray, uint8, float32

from .types import fileinput, choice


def load_image(filepath):
    img = Image.open(filepath).convert("RGBA")
    return (asarray(img, dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)


def to_pil_image(array):
    srgb = rint(array * 255).clip(0, 255).astype(uint8)
    return Image.fromarray(srgb.transpose(1, 2, 0), "RGBA")


def save_image(array, filepath):
    to_pil_image(array).save(filepath)


def clamp(a, x, b):
    return min(max(x, a), b)


def lerp(a, b, t):
    return a + t * (b - a)


def ilerp(a, b, x):
    return (x - a) / (b - a)


def weighted_median(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]


def chunks(length, count):
    div, mod = divmod(length, count)
    start = 0
    for i in range(count):
        stop = start + div + (1 if i < mod else 0)
        yield start, stop
        start = stop


def main():
    exit_code = 0
    parser = ArgumentParser(prog="eqbr", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Halftone Converter: an image converter to generate halftone images")
    parser.add_argument("input", metavar="FILE", type=fileinput, help="describe input image files (pass '-' to specify stdin)")
    parser.add_argument("output", metavar="FILE", type=fileinput, nargs="?", help="describe input image files (pass '-' to specify stdin)")
    parser.add_argument("-v", "--version", action="version", version="1")
    parser.add_argument("-F", "--resample", type=choice, choices=["nearest", "linear", "lanczos2", "lanczos3", "spline36"], default="linear", help="resampling method for determining dot size")
    args = parser.parse_args()

    f = args.input
    o = args.output
    n = 4
    m = 10

    i = load_image(f)

    print("Input:", i.shape)
    z = p(i, "", n)
    z = p(z.transpose((0, 2, 1)), "", m)
    z = z.transpose((0, 2, 1))
    save_image(z, sys.argv[2])


def p(x, mode, n):
    assert x.ndim == 3
    assert x.shape[0] == 4
    rgb = x[:3]
    alpha = x[3]
    brightness = np.max(rgb, axis=0)
    y = x.copy()
    hs = list(chunks(x.shape[2], n))
    for ((i1, i2), (ix, i3)) in zip(hs[:-1], hs[1:]):
        print(i1, i2, i3)
        assert i2 == ix
        b1 = weighted_median(brightness[:, i1:i2].ravel(), weights=alpha[:, i1:i2].ravel())
        b2 = weighted_median(brightness[:, i2:i3].ravel(), weights=alpha[:, i2:i3].ravel())
        # b1 = np.average(brightness[:, i1:i2], weights=alpha[:, i1:i2])
        # b2 = np.average(brightness[:, i2:i3], weights=alpha[:, i2:i3])
        c1 = i1 + (i2 - i1) // 2
        c2 = i2 + (i3 - i2) // 2
        edge1 = i1 == 0
        edge2 = i3 == x.shape[2]
        k1 = i1 if edge1 else c1
        k2 = i3 if edge2 else c2
        ts = np.linspace(start=(-0.5 if edge1 else 0.0), stop=(1.5 if edge2 else 1.0), num=(k2 - k1)).reshape((1, k2 - k1))
        a1 = np.zeros((1, k2 - k1)) + 0.0  # TODO
        a2 = b1 - b2  # TODO
        bias = lerp(a1, a2, ts)
        y[:, :, k1:k2] = x[:, :, k1:k2] + bias.reshape((1, 1, k2 - k1))
    return y
