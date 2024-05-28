from sys import float_info
from math import isfinite


def uint(string):
    value = int(string)
    if value >= 0:
        return value
    raise ValueError()


def positive(str):
    value = float(str)
    if isfinite(value) and value >= float_info.epsilon:
        return value
    else:
        raise ValueError()


def rate(str):
    value = float(str)
    if 0 <= value <= 1:
        return value
    else:
        raise ValueError()


def nonempty(str):
    if str:
        return str
    else:
        raise ValueError()


def fileinput(str):
    # stdin (-) を None で返す
    if str == "-":
        return None
    return nonempty(str)


def choice(label):
    return str.lower(label)
