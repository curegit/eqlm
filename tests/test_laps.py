import itertools
import numpy as np
from unittest import TestCase
from eqlm.laps.core import NamedStencil, sharpening_kernel

class KernelTest(TestCase):
    def test_simple5(self):
        self.assertTrue(np.all(NamedStencil.Simple5.value.array ==
            np.array(
          [[0,-1,0],[-1,4,-1],[0,-1,0]],
          dtype=np.float32,)
                               ))
    def test_simple9(self):
        self.assertTrue(np.allclose(NamedStencil.Simple9.value.array, np.array([[-1,  -1, -1],
           [-1, 8, -1],
           [-1,  -1, -1]], dtype=float) / 3) )


    def test_sharpening_kernel(self):
        for s, c in itertools.product(NamedStencil, [0.0, 0.5, 1.0, 2.5]):
            with self.subTest(stencil=s, coef=c):
                k = sharpening_kernel(s.value, coef=c)
                self.assertTrue( np.allclose( k.sum(), 1.0))

