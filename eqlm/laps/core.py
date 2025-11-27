import functools
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from numpy import ndarray
from ..img import Color


@dataclass(frozen=True, kw_only=True)
class ColorMode:
    color: Color
    channels: list[int]
    min: list[float] | float = 0.0
    max: list[float] | float = 1.0


class Mode(Enum):
    RGB = ColorMode(color=Color.RGB, channels=[0, 1, 2])
    Red = ColorMode(color=Color.RGB, channels=[0])
    Green = ColorMode(color=Color.RGB, channels=[1])
    Blue = ColorMode(color=Color.RGB, channels=[2])
    LAB = ColorMode(color=Color.LAB, channels=[0, 1, 2], min=[0.0, -127.0, -127.0], max=[100.0, 127.0, 127.0])
    AB = ColorMode(color=Color.LAB, channels=[1, 2], min=-127.0, max=127.0)
    L = ColorMode(color=Color.LAB, channels=[0], min=0.0, max=100.0)

    def __str__(self) -> str:
        if self.name == self.value.color.name:
            return self.name
        else:
            return f"{self.name} ({self.value.color.name})"


@dataclass(kw_only=True, slots=True)
class NinePointStencil:
    array: ndarray
    gamma: float

    def __init__(self, *, gamma: float = (1 / 2)):
        self.array = NinePointStencil.nine_point_stencil(gamma)
        self.gamma = gamma

    def __eq__(self, other):
        return isinstance(other, NinePointStencil) and self.gamma == other.gamma

    def __hash__(self):
        return hash(self.gamma)

    @staticmethod
    @functools.cache
    def nine_point_stencil(gamma):
        array = ((1.0 - gamma) * np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float) + gamma * np.array([[-1 / 2, 0, -1 / 2], [0, 2, 0], [-1 / 2, 0, -1 / 2]], dtype=float)).astype(np.float32).copy()
        array.flags.writeable = False
        return array


class NamedStencil(Enum):
    Basic5 = NinePointStencil(gamma=0.0)
    Basic9 = NinePointStencil(gamma=(2 / 3))
    Diagonal = NinePointStencil(gamma=1.0)
    OonoPuri = NinePointStencil(gamma=(1 / 2))
    PatraKarttunen = NinePointStencil(gamma=(1 / 3))

    def __str__(self) -> str:
        return self.name

    @property
    def description(self):
        match self:
            case NamedStencil.Basic5:
                return "typical 4-neighbor kernel"
            case NamedStencil.Basic9:
                return "typical 8-neighbor kernel"
            case NamedStencil.Diagonal:
                return "4-diagonal-neighbor kernel"
            case NamedStencil.OonoPuri:
                return "Oono-Puri isotropic kernel, reduced overall error"
            case NamedStencil.PatraKarttunen:
                return "Patra-Karttunen isotropic kernel, optimized for rotational invariance"
        return str(self)


modes = {m.name.lower(): m for m in Mode}
stencils = {s.name.lower(): s for s in NamedStencil}


identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)


def sharpening_kernel(stencil: NinePointStencil, *, coef: float = 1.0):
    return identity + (coef * stencil.array).astype(np.float32)


def laplacian_sharpening(x: ndarray, stencil: NinePointStencil, channels: list[int], *, coef: float = 0.2, clip: tuple[float, float] | list[tuple[float, float]] | None = None):
    kernel = sharpening_kernel(stencil=stencil, coef=coef)
    result = x.copy()
    result[:, :, channels] = cv2.filter2D(h := x[:, :, channels], -1, kernel, borderType=cv2.BORDER_REFLECT).reshape(h.shape)
    if clip is not None:
        if isinstance(clip, tuple):
            result = result.clip(*clip)
        else:
            for i, bound in zip(range(result.shape[2]), clip):
                result[:, :, i] = result[:, :, i].clip(*bound)
    return result
