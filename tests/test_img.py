import io
import numpy as np
from unittest import TestCase
from PIL import Image
from eqlm import img as eqimg
from eqlm.utils import filerelpath


class ImageCodecTest(TestCase):
    def test_read_png48(self):
        a, _ = eqimg.load_image(filerelpath("grad16.png"), normalize=False)
        self.assertEqual(a.dtype, np.uint16)

    def test_write_png48(self):
        i, _ = eqimg.load_image(filerelpath("exif.jpg"), normalize=True)
        buf = io.BytesIO()
        eqimg.save_image(i, buf, prefer16=True)
        a, _ = eqimg.load_image(buf.getbuffer(), normalize=False)
        self.assertEqual(a.dtype, np.uint16)


class ExifTest(TestCase):
    def test_exif_orientation(self):
        path = filerelpath("exif.jpg")
        i = Image.open(path)
        a, _ = eqimg.load_image(path, normalize=True, orientation=True)
        h, w, c = a.shape
        self.assertEqual(c, 3)
        self.assertEqual(h, i.width)
        self.assertEqual(w, i.height)
        i.close()


class ColorProfileTest(TestCase):
    def test_read_icc(self):
        _, icc = eqimg.load_image(filerelpath("icc.png"), normalize=True)
        self.assertIsInstance(icc, bytes)
        if isinstance(icc, bytes):
            self.assertGreater(len(icc), 0)

    def test_embed_icc(self):
        i, icc = eqimg.load_image(filerelpath("icc.png"), normalize=True)
        buf = io.BytesIO()
        eqimg.save_image(i, buf, icc_profile=icc)
        png = buf.getvalue()
        _, icc_emb = eqimg.load_image(png)
        self.assertIsInstance(icc_emb, bytes)
        if isinstance(icc_emb, bytes):
            self.assertGreater(len(icc_emb), 0)
            self.assertEqual(icc, icc_emb)
