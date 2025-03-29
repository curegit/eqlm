import numpy as np
from enum import Enum
from dataclasses import dataclass
from numpy import ndarray
from scipy.interpolate import BSpline
from scipy.interpolate import CubicSpline
from .img import Color
from .utils import lerp, chunks, weighted_median


@dataclass(frozen=True, kw_only=True)
class ColorMode:
    color: Color
    channel: int
    min: float = 0.0
    max: float = 1.0


class Mode(Enum):
    Luminance = ColorMode(color=Color.LAB, channel=0, min=0.0, max=100.0)
    Brightness = ColorMode(color=Color.HSV, channel=2)
    Saturation = ColorMode(color=Color.HSV, channel=1)
    Lightness = ColorMode(color=Color.HLS, channel=1)


modes = {m.name.lower(): m for m in Mode}


def biprocess(x: ndarray, n: tuple[int | None, int | None] = (2, 2), *, alpha: ndarray | None = None, target: float | None = None, median: bool = False, clamp: bool = False, clip: tuple[float, float] | None = None) -> ndarray:
    k, l = n
    weights = np.ones_like(x) if alpha is None else alpha
    z = process(x, weights, l, target=target, median=median, clamp=clamp, clip=clip) if l is not None and l >= 2 else x
    return process(z.transpose(1, 0), weights.transpose(1, 0), k, target=target, median=median, clamp=clamp, clip=clip).transpose((1, 0)) if k is not None and k >= 2 else z


def process(x: ndarray, w: ndarray, n: int = 2, *, target: float | None = None, median: bool = False, clamp: bool = False, clip: tuple[float, float] | None = None, use_spline: bool = True) -> ndarray:
    assert n >= 2
    assert x.ndim == w.ndim == 2
    if x.shape[1] < n:
        raise ValueError("Too many divisions")
    dest = np.zeros_like(x)
    divs = list(chunks(x.shape[1], n))

    def aggregate(x, w):
        w = w + 1e-8
        if median:
            return weighted_median(x.ravel(), weights=w.ravel())
        else:
            return np.average(x, weights=w)


    vs = []
    #for i, ((i1, i2), (_, i3)) in divpairs:
    #    if i == 0:
    #        v1 = aggregate(x[:, i1:i2], w[:, i1:i2])
    #        vs.append(v1)
    #    v2 = aggregate(x[:, i2:i3], w[:, i2:i3])
    #    vs.append(v2)
    for i1, i2 in divs:
        v = aggregate(x[:, i1:i2], w[:, i1:i2])
        vs.append(v)

    vt = np.mean(vs) if target is None else lerp(np.min(vs), np.max(vs), target)

    spline = None
    if use_spline:
        spline = CubicSpline(np.linspace(0, len(vs) - 1, len(vs)), vs)

    divpairs = list(enumerate(zip(divs[:-1], divs[1:])))
    for i, ((i1, i2), (_, i3)) in divpairs:
        c1 = i1 + (i2 - i1) // 2
        c2 = i2 + (i3 - i2) // 2
        edge1 = i1 == 0
        edge2 = i3 == x.shape[1]
        k1 = i1 if edge1 else c1
        k2 = i3 if edge2 else c2

        if spline is not None:
            t1 = float(i) - 0.5 if edge1 else float(i)
            t2 = float(i + 1) + 0.5 if edge2 else float(i + 1)
            ts = np.linspace(start=t1, stop=t2, num=(k2 - k1), endpoint=False)
            ys = spline(ts)
            print(ts[0], ts[-1])
            print(ys[0], ys[-1])
            y = x[:, k1:k2] - ys.reshape((1, k2 - k1)) + vt


        else:
            v1 = vs[i]
            v2 = vs[i + 1]
            ts = np.linspace(start=(-0.5 if edge1 else 0.0), stop=(1.5 if edge2 else 1.0), num=(k2 - k1)).reshape((1, k2 - k1))
            if clamp:
                ts = ts.clip(0.0, 1.0)
            grad = lerp(0.0, v1 - v2, ts)
            bias = vt - v1
            y = x[:, k1:k2] + grad.reshape((1, k2 - k1)) + bias



        dest[:, k1:k2] = y if clip is None else y.clip(*clip)
    return dest
