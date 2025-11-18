import functools
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from numpy import ndarray


@dataclass( kw_only=True)
class NinePointStencil:
    array: ndarray
    gamma: float

    def __init__(self, *, gamma:float=(1/2)):
        self.array = NinePointStencil.nine_point_stencil(gamma)
        self.gamma = gamma

    def __eq__(self, other):
        return isinstance(other, NinePointStencil) and self.gamma == other.gamma

    def __hash__(self):
        return hash(self.gamma)

    @staticmethod
    @functools.cache
    def nine_point_stencil(gamma):
        array =  (
        (1.0 - gamma) * np.array(
            [[0,-1,0],[-1,4,-1],[0,-1,0]],
            dtype=float,
            ) + gamma * np.array(
                [[-1/2, 0, -1/2], [0,2,0], [-1/2, 0, -1/2]], dtype=float,
            )
        ).astype(np.float32).copy()
        array.flags.writeable = False
        return array


class NamedStencil(Enum):
     Simple5 = NinePointStencil(gamma=0.0)
     Simple9 = NinePointStencil(gamma=(2/3))
     Diagonal = NinePointStencil(gamma=1.0)
     OonoPuri = NinePointStencil(gamma=(1/2))
     PatraKarttunen = NinePointStencil(gamma=(1/3))

stencils = {s.name.lower(): s for s in NamedStencil}


identity = np.array([[0,  0, 0],
          [0, 1, 0],
          [0, 0, 0]], dtype=np.float32)


def sharpening_kernel(stencil: NinePointStencil, *, coef:float=1.0):
     return  identity + (coef * stencil.array).astype(np.float32)


def laplacian_sharpening(x: ndarray, stencil: NinePointStencil, *,coef:float=1.0, clip: tuple[float, float] | None = None):
    kernel = sharpening_kernel(stencil=stencil, coef=coef)
    result = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return result if clip is None else result.clip(*clip)
