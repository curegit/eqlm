import sys
import contextlib
from time import time
from glob import glob
from os.path import isfile
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image, ImageCms
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn


import numpy as np

from sys import float_info
from os.path import basename
from math import isfinite
from PIL import ImageCms

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

# ファイル名の一部を受け入れる変換関数
def filenameseg(str):
	if str == basename(str):
		return str
	else:
		raise ValueError()

# 大小文字を区別しないラベルマッチのための変換関数
def choice(label):
	return str.lower(label)

# ラベルを受け入れてレンダリングインテントを表す整数を返す
def intent(label):
	label = str.lower(label)
	if label == "per":
		return ImageCms.Intent.PERCEPTUAL
	if label == "sat":
		return ImageCms.Intent.SATURATION
	if label == "rel":
		return ImageCms.Intent.RELATIVE_COLORIMETRIC
	if label == "abs":
		return ImageCms.Intent.ABSOLUTE_COLORIMETRIC
	if 0 <= int(label) <= 3:
		return int(label)
	else:
		raise ValueError()












from PIL import Image
from PIL.Image import Resampling
from numpy import rint, asarray, uint8, float32

def load_image(filepath):
	img = Image.open(filepath).convert("RGBA")
	return (asarray(img, dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)

def load_image_uint8(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Resampling.LANCZOS)
	return asarray(img, dtype=uint8).transpose(2, 0, 1)

def uint8_to_float(array):
	return (array / 255).astype(float32)

def to_pil_image(array):
	srgb = rint(array * 255).clip(0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGBA")

def save_image(array, filepath):
	to_pil_image(array).save(filepath)

def lerp(a, b, t):
	h = (b - a)
	print(h.shape)
	x = t * h
	print(x.shape)
	return a + x


def clamp(a, x, b):
	return min(max(x, a), b)

#def lerp(a, b, t):
#	return a + t * (b - a)

def ilerp(a, b, x):
	return (x - a) / (b - a)




def chunks(length, count):
    div, mod = divmod(length, count)
    start = 0
    for i in range(count):
        stop = start + div + (1 if i < mod else 0)
        yield start, stop
        start = stop



# メイン関数
def main():
	exit_code = 0


	# コマンドライン引数をパース
	parser = ArgumentParser(prog="eqbr", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="Halftone Converter: an image converter to generate halftone images")
	parser.add_argument("input", metavar="FILE", type=fileinput, help="describe input image files (pass '-' to specify stdin)")
	parser.add_argument("output", metavar="FILE", type=fileinput, nargs="?", help="describe input image files (pass '-' to specify stdin)")
	parser.add_argument("-v", "--version", action="version", version="1")
	parser.add_argument("-F", "--resample", type=choice, choices=["nearest", "linear", "lanczos2", "lanczos3", "spline36"], default="linear", help="resampling method for determining dot size")
	args = parser.parse_args()

	f = args.input
	o = args.output


	i = load_image(f)

	print(i.shape)


	print(brightness.shape)
	height, width = brightness.shape
	left_b = np.average(brightness[:,:width//2], weights=alpha[:,:width//2])
	right_b = np.average(brightness[:,width//2:], weights=alpha[:,width//2:])
	print(left_b, right_b)


	#diver = (right_b - left_b)
	ts1 = np.linspace(start=0.0, stop=1.0, num=width).reshape((1, width))
	ts2 = np.linspace(start=0.0, stop=1.0, num=height).reshape((height, 1))
	bias = lerp(np.zeros((height, width)), (left_b - right_b) * 2, ts1)
	i[:3] = i[:3] + bias.reshape((1, height, width))
	res = i
	save_image(res, sys.argv[2])
	return

	brightness = np.max(rgb, axis=0)

	rgb = i[:3]
	alpha = i[3]
	# 
	hs = list(chunks(width, 2))
	vs = list(chunks(height, 2))
	for i, ((h1, h2), (h3, h4)) in enumerate(zip(hs[:-1], hs[1:])):
		assert h2 == h3
		for j, ((v1, v2), (v3, v4)) in enumerate(zip(vs[:-1], vs[1:])):
			assert v2 == v3
			left = i == 0
			right = h4 == width
			top = j == 0
			bottom = j == height
			l = 0 if left else 
			r = #
			h = 
			match args.mode:
				case "brightness":
					
					
				case "saturation":
					
				case _:
					raise ValueError()


main()
