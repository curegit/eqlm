import io
import itertools
import numpy as np
from itertools import product
from contextlib import redirect_stderr
from unittest import TestCase
from eqlm.laps import core, main
from eqlm.laps.core import NamedStencil, sharpening_kernel
from eqlm.utils import filerelpath


class KernelTest(TestCase):
    def test_basic5(self):
        self.assertTrue(np.all(NamedStencil.Basic5.value.array == np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)))

    def test_basic9(self):
        self.assertTrue(np.allclose(NamedStencil.Basic9.value.array, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float) / 3))

    def test_sharpening_kernel(self):
        for s, c in itertools.product(NamedStencil, [0.0, 0.5, 1.0, 2.5]):
            with self.subTest(stencil=s, coef=c):
                k = sharpening_kernel(s.value, coef=c)
                self.assertTrue(np.allclose(k.sum(), 1.0))


class MainTest(TestCase):
    def test_process(self):
        fpaths = list(map(filerelpath, ["tsurumai.webp", "p3.tiff", "ball.png"]))
        for input_file, mode, stencil, include_alpha, gamma in product(fpaths, core.Mode, core.stencils.values(), [True, False], [2.2, None]):
            with self.subTest(input_file=input_file, mode=mode, stencil=stencil, include_alpha=include_alpha, gamma=gamma):
                buf = io.BytesIO()
                with redirect_stderr(io.StringIO()):
                    main.laps(
                        input_file=input_file,
                        output_file=buf,
                        mode=mode,
                        stencil=stencil,
                        include_alpha=include_alpha,
                        gamma=gamma,
                    )
                self.assertTrue(buf.getvalue())
