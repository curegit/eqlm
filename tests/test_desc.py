import io
from itertools import product
from contextlib import redirect_stderr
from unittest import TestCase
from eqlm.desc import main
from eqlm.utils import filerelpath


class MainTest(TestCase):
    def test_process(self):
        fpaths = list(map(filerelpath, ["patch.png"]))
        for source_file, cmyk, nl_means in product(fpaths, [True, False], [True, False]):
            with self.subTest(source_file=source_file, cmyk=cmyk, nl_means=nl_means):
                buf = io.BytesIO()
                with redirect_stderr(io.StringIO()):
                    main.desc(
                        input_file=source_file,
                        output_file=buf,
                        cmyk=cmyk,
                        nl_means=nl_means,
                    )
                self.assertTrue(buf.getvalue())
