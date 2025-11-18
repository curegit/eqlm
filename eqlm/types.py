import os
from sys import float_info
from pathlib import Path
from math import isfinite
from .utils import alt_filepath


def uint(string: str):
    value = int(string)
    if value >= 0:
        return value
    raise ValueError()


def real(string: str):
    value = float(string)
    if isfinite(value):
        return value
    raise ValueError()


def ufloat(string: str):
    value = real(string)
    if value >= 0:
        return value
    raise ValueError()


def positive(string: str):
    value = real(string)
    if value >= float_info.epsilon:
        return value
    raise ValueError()


def rate(string: str):
    value = float(string)
    if 0 <= value <= 1:
        return value
    else:
        raise ValueError()


def nonempty(string: str):
    if string:
        return string
    else:
        raise ValueError()


def fileinput(string: str):
    # stdin (-) を None で返す
    if string == "-":
        return None
    # clipboard (_)
    if string == "_":
        return Clipboard()
    path = Path(nonempty(string)).resolve(strict=True)
    if path.is_file():
        return path
    else:
        raise FileNotFoundError(f"No such file: {path}")


def fileoutput(string: str):
    # stdout (-) を None で返す
    if string == "-":
        return None
    # clipboard (_)
    if string == "_":
        return Clipboard()
    path = Path(nonempty(string)).resolve()
    if path.exists(follow_symlinks=False):
        if path.is_file():
            return path
        else:
            raise RuntimeError(f"Path already exists: {path}")
    else:
        if path.parent.is_dir():
            return path
        else:
            raise RuntimeError(f"Destination directory doesn't exist: {path.parent}")


def choice(label: str):
    return str.lower(label)


class Clipboard:
    def __str__(self) -> str:
        return "Clipboard"


class AutoUniquePath:
    def __str__(self) -> str:
        return "AutoUnique"

    def __init__(self, *, input_path: Path | str | None = None, suffix: str = ""):
        self.input_path = input_path
        self.suffix = suffix

    def open(self, *, input_path: Path | str | None = None, suffix: str | None = None, ext: str = ""):
        if input_path is None:
            input_path = self.input_path
        if input_path is None:
            raise RuntimeError("Input path is not specified")
        if suffix is None:
            suffix = self.suffix
        path = Path(input_path).resolve(strict=False)
        filepath = (Path(".") / (path.stem + suffix)).with_suffix(f"{os.extsep}{ext}" if ext else "")
        while True:
            try:
                return open(filepath, "xb"), filepath
            except FileExistsError:
                filepath = alt_filepath(filepath)

    def open_png(self, *, input_path: Path | str | None = None, suffix: str | None = None):
        return self.open(input_path=input_path, suffix=suffix, ext="png")
