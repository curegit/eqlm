import io
import numpy as np
from itertools import product
from contextlib import redirect_stderr
from unittest import TestCase
from eqlm import img as eqimg
from eqlm.match import core, main
from eqlm.utils import filerelpath


class MainTest(TestCase):
    def test_process(self):
        fpaths = list(map(filerelpath, ["tsurumai.webp", "p3.tiff", "ball.png"]))
        for source_file, reference_file, mode, gamma, alpha in product(fpaths, fpaths, core.Mode, [2.2, None], [(0.0, 0.5), (None, None)]):
            with self.subTest(source_file=source_file, reference_file=reference_file, mode=mode, gamma=gamma):
                buf = io.BytesIO()
                with redirect_stderr(io.StringIO()):
                    main.match(
                        source_file=source_file,
                        reference_file=reference_file,
                        output_file=buf,
                        mode=mode,
                        alpha=alpha,
                        gamma=gamma,
                    )
                self.assertTrue(buf.getvalue())

    def test_identity(self):
        fpath = filerelpath("tsurumai.webp")
        for mode, slow in product(core.Mode, [True, False]):
            with self.subTest(mode=mode, slow=slow):
                buf = io.BytesIO()
                with redirect_stderr(io.StringIO()):
                    main.match(
                        source_file=fpath,
                        reference_file=fpath,
                        output_file=buf,
                        mode=mode,
                        slow=slow,
                        orientation=True,
                    )
                res, _ = eqimg.load_image(buf.getbuffer(), normalize=True, orientation=True)
                x, _ = eqimg.load_image(fpath, normalize=True, orientation=True)
                self.assertEqual(res.shape, x.shape)
                self.assertTrue(np.allclose(res, x, rtol=0, atol=(1 / (2**8 - 1))))
