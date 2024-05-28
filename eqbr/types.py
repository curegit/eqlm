from sys import float_info
from pathlib import Path
from math import isfinite


def uint(string: str):
    value = int(string)
    if value >= 0:
        return value
    raise ValueError()


def positive(string: str):
    value = float(string)
    if isfinite(value) and value >= float_info.epsilon:
        return value
    else:
        raise ValueError()


def rate(string: str):
    value = float(string)
    if 0 <= value <= 1:
        return value
    else:
        raise ValueError()


def nonempty(string:str):
    if string:
        return string
    else:
        raise ValueError()


def fileinput(string:str):
    # stdin (-) を None で返す
    if string == "-":
        return None
    p = Path(nonempty(string)).resolve(strict=True)
    if p.is_file():
        return p
    else:
        raise RuntimeError(f"No such file: {p}")


def fileoutput(string:str):
    # stdout (-) を None で返す
    if string == "-":
        return None
    p = Path(nonempty(string))
    if p.exists():
        if p.is_file():
            return p
        else:
            raise RuntimeError(f"Path already exists: {p}")
    else:
        return p


def choice(label:str):
    return str.lower(label)
