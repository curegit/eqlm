from sys import float_info
from os.path import basename
from math import isfinite


def uint(string):
    value = int(string)
    if value >= 0:
        return value
    raise ValueError()


# 正の実数を受け入れる変換関数
def positive(str):
    value = float(str)
    if isfinite(value) and value >= float_info.epsilon:
        return value
    else:
        raise ValueError()


# 0-1 の実数を受け入れる変換関数
def rate(str):
    value = float(str)
    if 0 <= value <= 1:
        return value
    else:
        raise ValueError()


# 非空列を受け入れる変換関数
def nonempty(str):
    if str:
        return str
    else:
        raise ValueError()


# 入力ファイルパスを受け入れる変換関数
def fileinput(str):
    # stdin (-) を None で返す
    if str == "-":
        return None
    return nonempty(str)


# 大小文字を区別しないラベルマッチのための変換関数
def choice(label):
    return str.lower(label)
