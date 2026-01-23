import io
import numpy as np
from itertools import product
from contextlib import redirect_stderr
from unittest import TestCase
from eqlm import img as eqimg
from eqlm.eq import core, main
from eqlm.img import color_transforms
from eqlm.utils import filerelpath


class MainTest(TestCase):
    def test_process(self):
        for mode, vertical, horizontal, interpolation, target, median, clamp, deep in product(core.Mode, [2, 4], [2, 3], core.Interpolation, [0.0, 0.5, 1.0], [True, False], [True, False], [True, False]):
            with self.subTest(mode=mode, vertical=vertical, horizontal=horizontal, interpolation=interpolation, target=target, median=median, clamp=clamp, deep=deep):
                buf = io.BytesIO()
                with redirect_stderr(io.StringIO()):
                    main.equalize(
                        input_file=filerelpath("icc.png"),
                        output_file=buf,
                        mode=mode,
                        vertical=vertical,
                        horizontal=horizontal,
                        interpolation=interpolation,
                        target=target,
                        clamp=clamp,
                        median=median,
                        deep=deep,
                    )
                self.assertTrue(buf.getvalue())


class CoreTest(TestCase):
    def test_biprocess(self):
        for mode, vertical, horizontal, interpolation, target, median, clamp in product(core.Mode, [2, 4], [2, 3], core.Interpolation, [0.0, 0.5, 1.0], [True, False], [True, False]):
            with self.subTest(mode=mode, vertical=vertical, horizontal=horizontal, interpolation=interpolation, target=target, median=median, clamp=clamp):
                x, _ = eqimg.load_image(filerelpath("tsurumai.webp"), normalize=True, orientation=True)
                f, g = color_transforms(mode.value.color, transpose=True)
                v = f(x)
                c = mode.value.channel
                v[c] = core.biprocess(v[c], n=(vertical, horizontal), interpolation=(interpolation, interpolation), target=target, median=median, clamp=clamp, clip=(mode.value.min, mode.value.max))
                y = g(v)
                self.assertEqual(x.shape, y.shape)

    def test_identity(self):
        for orientation in [True, False]:
            with self.subTest(orientation=orientation):
                x, _ = eqimg.load_image(filerelpath("tsurumai.webp"), normalize=True, orientation=orientation)
                mode = core.Mode.Brightness
                f, g = color_transforms(mode.value.color, transpose=True, gamma=None)
                v = f(x)
                c = mode.value.channel
                v[c] = core.biprocess(v[c], n=(None, None), clip=(mode.value.min, mode.value.max))
                y = g(v)
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(np.allclose(x, y, rtol=0, atol=(1 / (2**8 - 1) / 2)))
