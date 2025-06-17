import numpy as np
from enum import Enum
from dataclasses import dataclass
from numpy import ndarray
from skimage.exposure import match_histograms
from ..img import Color


@dataclass(frozen=True, kw_only=True)
class ColorMode:
    color: Color
    channels: list[int]


class Mode(Enum):
    RGB = ColorMode(color=Color.RGB, channels=[0, 1, 2])
    Red = ColorMode(color=Color.RGB, channels=[0])
    Green = ColorMode(color=Color.RGB, channels=[1])
    Blue = ColorMode(color=Color.RGB, channels=[2])
    LAB = ColorMode(color=Color.LAB, channels=[0, 1, 2])
    AB = ColorMode(color=Color.LAB, channels=[1, 2])
    L = ColorMode(color=Color.LAB, channels=[0])
    Brightness = ColorMode(color=Color.HSV, channels=[2])
    Saturation = ColorMode(color=Color.HSV, channels=[1])
    Lightness = ColorMode(color=Color.HLS, channels=[1])

    def __str__(self) -> str:
        if self.name == self.value.color.name:
            return self.name
        else:
            return f"{self.name} ({self.value.color.name})"


modes = {m.name.lower(): m for m in Mode}


def histogram_matching(x: ndarray, r: ndarray, channels: list[int], *, x_alpha: ndarray | None = None, r_alpha: ndarray | None = None, x_alpha_threshold: float = 0.0, r_alpha_threshold: float = 0.5):
    assert x.ndim == r.ndim == 3
    assert x_alpha is None or x_alpha.ndim == 2
    assert r_alpha is None or r_alpha.ndim == 2
    dest = x.copy()
    for channel in channels:
        x_channel = x[channel]
        r_channel = r[channel]
        if x_alpha is not None:
            assert x_alpha.shape == x_channel.shape
            x_mask = x_alpha > x_alpha_threshold
        else:
            x_mask = np.ones_like(x_channel, dtype=bool)
        if r_alpha is not None:
            assert r_alpha.shape == r_channel.shape
            r_mask = r_alpha > r_alpha_threshold
        else:
            r_mask = np.ones_like(r_channel, dtype=bool)
        if x_mask.sum() == 0:
            raise ValueError("No pixels to match in source channel (all pixels are considered transparent)")
        if r_mask.sum() == 0:
            raise ValueError("No pixels to match in reference channel (all pixels are considered transparent)")
        a = x_channel[x_mask].reshape((1, -1))
        b = r_channel[r_mask].reshape((1, -1))
        matched = match_histograms(a, b, channel_axis=None)
        dest[channel][x_mask] = matched.ravel()
    return dest
