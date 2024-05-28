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


def biprocess(x: ndarray, n: tuple[int | None, int | None] = (2, 2), *, alpha: ndarray|None = None, target: float | None = None, median: bool = False, clip: tuple[float, float] | None = None) -> ndarray:
    k, l = n
    weights = np.ones_like(x) if alpha is None else alpha
    z = process(x, weights, l, target=target, median=median, clip=clip) if l is not None and l >= 2 else x
    return process(z.transpose(1, 0), weights.transpose(1, 0), k, target=target, median=median, clip=clip).transpose((1, 0)) if k is not None and k >= 2 else z


def process(x: ndarray, w: ndarray, n: int = 2, *, target: float | None = None, median: bool = False, clip: tuple[float, float] | None = None) -> ndarray:
    assert n >= 2
    assert x.ndim == w.ndim == 2

    y = np.zeros_like(x)
    hs = list(chunks(x.shape[1], n))

    def aggregate(x, w):
        if median:
            return weighted_median(x.ravel(), weights=w.ravel())
        else:
            return np.average(x, weights=w)

    def progress():
        pass

    s = list(enumerate(zip(hs[:-1], hs[1:])))

    bs = []
    for i, ((i1, i2), (ix, i3)) in s:
        if i == 0:
            b1 = aggregate(x[:, i1:i2], w[:, i1:i2])
            bs.append(b1)
        b2 = aggregate(x[:, i2:i3], w[:, i2:i3])
        bs.append(b2)

    bg = np.mean(bs) if target is None else lerp(np.min(bs), np.max(bs), target)
    for i, ((i1, i2), (ix, i3)) in s:
        b1 = bs[i]
        b2 = bs[i + 1]
        c1 = i1 + (i2 - i1) // 2
        c2 = i2 + (i3 - i2) // 2
        edge1 = i1 == 0
        edge2 = i3 == x.shape[1]
        k1 = i1 if edge1 else c1
        k2 = i3 if edge2 else c2
        ts = np.linspace(start=(-0.5 if edge1 else 0.0), stop=(1.5 if edge2 else 1.0), num=(k2 - k1)).reshape((1, k2 - k1))
        grad = lerp(0.0, b1 - b2, ts)
        bias = bg - b1
        r = x[:, k1:k2] + grad.reshape((1, k2 - k1)) + bias
        y[:, k1:k2] = r if clip is None else r.clip(*clip)
    return y
