from unittest import TestCase
from eqlm import utils


class IterTest(TestCase):
    def test_chunks(self):
        self.assertEqual(list(utils.chunks(8, 2)), [(0, 4), (4, 8)])
        self.assertEqual(list(utils.chunks(10, 3)), [(0, 4), (4, 7), (7, 10)])
