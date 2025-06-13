from enum import Enum
from dataclasses import dataclass
from numpy import ndarray
from skimage.exposure import match_histograms
from ..img import Color


@dataclass(frozen=True, kw_only=True)
class ColorMode:
    color: Color
    channels: list[int]
    min: float = 0.0
    max: float = 1.0


class Mode(Enum):
    RGB = ColorMode(color=Color.RGB, channels=[0, 1, 2])
    Red = ColorMode(color=Color.RGB, channels=[0])
    Green = ColorMode(color=Color.RGB, channels=[1])
    Blue = ColorMode(color=Color.RGB, channels=[2])
    LAB = ColorMode(color=Color.LAB, channels=[0, 1, 2], min=0.0, max=100.0)
    AB = ColorMode(color=Color.LAB, channels=[1, 2], min=0.0, max=100.0)
    Luminance = ColorMode(color=Color.LAB, channels=[0], min=0.0, max=100.0)
    Brightness = ColorMode(color=Color.HSV, channels=[2])
    Saturation = ColorMode(color=Color.HSV, channels=[1])
    Lightness = ColorMode(color=Color.HLS, channels=[1])


modes = {m.name.lower(): m for m in Mode}


def histgram_matching(x: ndarray, r: ndarray, channels: list[int]):
    dest = x.copy()
    for channel in channels:
        matched = match_histograms(x[channel], r[channel], channel_axis=None)
        dest[channel] = matched
    return dest
