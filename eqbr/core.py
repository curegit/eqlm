import sys
import contextlib
from time import time
from glob import glob
from os.path import isfile
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image

# from rich.console import Console
# from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

from io import BufferedIOBase

import numpy as np
from PIL import Image
from numpy import rint, asarray, uint8, float32, ndarray

from pathlib import Path

from .types import fileinput, choice

import cv2

from enum import Enum


class Color(Enum):
    HSV = cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR
    HLS = cv2.COLOR_BGR2HLS, cv2.COLOR_HLS2BGR
    LAB = cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR

from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class ColorOp:
    color: Color
    channel: int

class Mode(Enum):
    Brightness = ColorOp(color=Color.HSV, channel=2)
    Saturation = ColorOp(color=Color.HSV, channel=1)
    Lightness = ColorOp(color=Color.HLS, channel=1)
    Luminance = ColorOp(color=Color.LAB, channel=0)

modes = {m.name.lower(): m for m in Mode}

def load_image(filelike, *, normalize: bool = True):
    match filelike:
        case str() | Path() as path:
            with open(Path(path).resolve(strict=True), "rb") as fp:
                buffer = fp.read()
        case bytes() as buffer:
            pass
        case _:
            raise ValueError()
    # OpenCV が ASCII パスしか扱えない問題を回避するためにバッファを経由する
    bin = np.frombuffer(buffer, np.uint8)
    #flags = cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH
    #if not orient:
    #    flags |= cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(bin, cv2.IMREAD_UNCHANGED)
    match img.dtype:
        case np.uint8:
            if normalize:
                return (img / (2**8 - 1)).astype(np.float32)
            else:
                return img
        case np.uint16:
            if normalize:
                return (img / (2**16 - 1)).astype(np.float32)
            else:
                return img
        case np.float32:
            return img
        case _:
            raise RuntimeError()

def save_image(img: ndarray, filelike: str | Path | BufferedIOBase, *, prefer16=False) -> None:
    match img.dtype:
        case np.float32:
            if prefer16:
                qt = 2**16 - 1
                dtype = np.uint16
            else:
                qt = 2**8 - 1
                dtype = np.uint8
            arr = np.rint(img * qt).clip(0, qt).astype(dtype)
        case np.uint8 | np.uint16:
            arr = img
        case _:
            raise ValueError()
    ok, bin = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError()
    buffer = bin.tobytes()
    match filelike:
        case str() | Path() as path:
            with open(Path(path), "wb") as fp:
                fp.write(buffer)
        case BufferedIOBase() as stream:
            stream.write(buffer)
        case _:
            raise ValueError()


def color_transforms(color: Color, *, gamma:float|None=2.2, transpose=False):
    f, r = color.value
    def g(x):
        y = cv2.cvtColor(x if gamma is None else x ** gamma, f)
        return y.transpose(2, 0, 1) if transpose else y
    def h(x):
        z = cv2.cvtColor(x.transpose(1, 2, 0) if transpose else x, r)
        return z if gamma is None else z ** (1 / gamma)
    return g, h

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

def split_alpha(x):
    if x.shape[2] == 4:
        return x[:, :, :3], x[:, :, 3]
    elif x.shape[2] == 3:
        return x, None
    raise ValueError()

def merge_alpha(x, a=None):
    if a is None:
        return x
    return np.concatenate((x, a[:, :, np.newaxis]), axis=2)

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
    n = 3 # w
    m = 2 # h

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


def bi_pp(x, channel:int=0, n:tuple[int | None, int | None]=(2, 2), *, alpha=None):
    k, l = n
    weights = np.ones_like(x[channel]) if alpha is None else alpha
    
    z = p(x, weights, channel, l) if l else x
    return p(z.transpose(0, 2, 1), weights.transpose(1, 0), channel, k).transpose((0, 2, 1)) if k else z

def p(x, w, channel=0, n=2):
    assert x.ndim == 3
    assert w.ndim == 2
    #rgb = x[:3]
    #alpha = x[3]
    #brightness = np.max(rgb, axis=0)
    y = x.copy()
    hs = list(chunks(x.shape[2], n))

    values = x[channel]


    
    grad: dict[tuple[int, int], tuple[float, ]] = {}

    g = weighted_median(values.ravel(), weights=w.ravel())
    for ((i1, i2), (ix, i3)) in zip(hs[:-1], hs[1:]):
        print(i1, i2, i3)
        assert i2 == ix
        b1 = weighted_median(values[:, i1:i2].ravel(), weights=w[:, i1:i2].ravel())
        b2 = weighted_median(values[:, i2:i3].ravel(), weights=w[:, i2:i3].ravel())
        # b1 = np.average(brightness[:, i1:i2], weights=alpha[:, i1:i2])
        # b2 = np.average(brightness[:, i2:i3], weights=alpha[:, i2:i3])
        c1 = i1 + (i2 - i1) // 2
        c2 = i2 + (i3 - i2) // 2
        edge1 = i1 == 0
        edge2 = i3 == x.shape[2]
        k1 = i1 if edge1 else c1
        k2 = i3 if edge2 else c2
        ts = np.linspace(start=(-0.5 if edge1 else 0.0), stop=(1.5 if edge2 else 1.0), num=(k2 - k1)).reshape((1, k2 - k1))
        #if b2 > b1:
        a1 = np.zeros((1, k2 - k1)) + (g - b1) # TODO
        a2 = b1 - b2 + (g - b1)  # TODO
        #else:
        #    a2 = np.zeros((1, k2 - k1)) + g # TODO
        #    a1 = b2 - b1 + g  # TODO
        #grad[(k1, k2)] = 
        bias = lerp(a1, a2, ts)
        y[channel, :, k1:k2] = x[channel, :, k1:k2] + bias.reshape((1, 1, k2 - k1))
    return y
