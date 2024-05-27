
# from rich.console import Console
# from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn


import numpy as np




from .utils import lerp, chunks, weighted_median

from enum import Enum




from dataclasses import dataclass

from .img import Color

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
