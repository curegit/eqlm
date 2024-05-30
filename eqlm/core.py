import numpy as np
from enum import Enum
from dataclasses import dataclass
from numpy import ndarray
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


def biprocess(x: ndarray, n: tuple[int | None, int | None] = (2, 2), *, alpha: ndarray | None = None, target: float | None = None, median: bool = False, clip: tuple[float, float] | None = None) -> ndarray:
    k, l = n
    weights = np.ones_like(x) if alpha is None else alpha
    z = process(x, weights, l, target=target, median=median, clip=clip) if l is not None and l >= 2 else x
    return process(z.transpose(1, 0), weights.transpose(1, 0), k, target=target, median=median, clip=clip).transpose((1, 0)) if k is not None and k >= 2 else z


def process(x: ndarray, w: ndarray, n: int = 2, *, target: float | None = None, median: bool = False, clip: tuple[float, float] | None = None) -> ndarray:
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

    divpairs = list(enumerate(zip(divs[:-1], divs[1:])))
    vs = []
    for i, ((i1, i2), (ix, i3)) in divpairs:
        if i == 0:
            v1 = aggregate(x[:, i1:i2], w[:, i1:i2])
            vs.append(v1)
        v2 = aggregate(x[:, i2:i3], w[:, i2:i3])
        vs.append(v2)
    vt = np.mean(vs) if target is None else lerp(np.min(vs), np.max(vs), target)
    for i, ((i1, i2), (ix, i3)) in divpairs:
        v1 = vs[i]
        v2 = vs[i + 1]
        c1 = i1 + (i2 - i1) // 2
        c2 = i2 + (i3 - i2) // 2
        edge1 = i1 == 0
        edge2 = i3 == x.shape[1]
        k1 = i1 if edge1 else c1
        k2 = i3 if edge2 else c2
        ts = np.linspace(start=(-0.5 if edge1 else 0.0), stop=(1.5 if edge2 else 1.0), num=(k2 - k1)).reshape((1, k2 - k1))
        grad = lerp(0.0, v1 - v2, ts)
        bias = vt - v1
        y = x[:, k1:k2] + grad.reshape((1, k2 - k1)) + bias
        dest[:, k1:k2] = y if clip is None else y.clip(*clip)
    return dest
