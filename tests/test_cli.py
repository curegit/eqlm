import io
from contextlib import redirect_stdout, redirect_stderr
from unittest import TestCase
from eqlm import cli


class CLITest(TestCase):

    subcommands = ["eq", "match", "laps", "desc"]

    def test_no_args(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(i := io.StringIO()):
                cli.main(argv=[])
        self.assertTrue(i.getvalue())

    def test_help(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(i := io.StringIO()):
                cli.main(argv=["-h"])
        self.assertTrue(i.getvalue())

    def test_version(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(i := io.StringIO()):
                cli.main(argv=["-v"])
        self.assertTrue(i.getvalue())

    def test_subcommand_no_args(self):
        for subcommand in self.subcommands:
            with self.subTest(subcommand=subcommand):
                with self.assertRaises(SystemExit):
                    with redirect_stderr(i := io.StringIO()):
                        cli.main(argv=[subcommand])
                self.assertTrue(i.getvalue())

    def test_subcommand_help(self):
        for subcommand in self.subcommands:
            with self.subTest(subcommand=subcommand):
                with self.assertRaises(SystemExit):
                    with redirect_stdout(i := io.StringIO()):
                        cli.main(argv=[subcommand, "-h"])
                self.assertTrue(i.getvalue())
