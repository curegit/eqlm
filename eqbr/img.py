import cv2
import numpy as np
from pathlib import Path
from enum import Enum
from io import BufferedIOBase
from numpy import ndarray


def load_image(filelike: str | Path | bytes | memoryview, *, normalize: bool = True) -> ndarray:
    match filelike:
        case str() | Path() as path:
            with open(Path(path).resolve(strict=True), "rb") as fp:
                buffer = fp.read()
        case bytes() | memoryview() as buffer:
            pass
        case _:
            raise ValueError()
    # OpenCV が ASCII パスしか扱えない問題を回避するためにバッファを経由する
    bin = np.frombuffer(buffer, np.uint8)
    # flags = cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH
    # if not orient:
    #    flags |= cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(bin, cv2.IMREAD_UNCHANGED)
    match img.dtype:
        case np.uint8:
            if normalize:
                return (img / (2**8 - 1)).astype(np.float32)
            else:
                return img
        case np.uint16:
            if normalize:
                return (img / (2**16 - 1)).astype(np.float32)
            else:
                return img
        case np.float32:
            return img
        case _:
            raise RuntimeError()


def save_image(img: ndarray, filelike: str | Path | BufferedIOBase, *, prefer16=False, icc_profile: bytes) -> None:
    match img.dtype:
        case np.float32:
            if prefer16:
                qt = 2**16 - 1
                dtype = np.uint16
            else:
                qt = 2**8 - 1
                dtype = np.uint8
            arr = np.rint(img * qt).clip(0, qt).astype(dtype)
        case np.uint8 | np.uint16:
            arr = img
        case _:
            raise ValueError()
    ok, bin = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError()
    buffer = bin.tobytes()

    # ICCプロファイルをPNGデータに追加
    icc_chunk = b'\x00\x00\x00' + len(icc_profile).to_bytes(4, 'big') + b'iCCP' + b'\x00' + b'\x00' + b'\x00' + icc_profile
    buffer = buffer[:33] + icc_chunk + buffer[33:]

    match filelike:
        case str() | Path() as path:
            with open(Path(path), "wb") as fp:
                fp.write(buffer)
        case BufferedIOBase() as stream:
            stream.write(buffer)
        case _:
            raise ValueError()


def get_icc():
    maybe_icc = image.info.get("icc_profile")
    if not prefer_embedded or maybe_icc is None:
        return ImageCms.applyTransform(image, transform)
    em_profile = ImageCms.ImageCmsProfile(io.BytesIO(maybe_icc))


class Color(Enum):
    HSV = cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR
    HLS = cv2.COLOR_BGR2HLS, cv2.COLOR_HLS2BGR
    LAB = cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR


def color_transforms(color: Color, *, gamma: float | None = 2.2, transpose: bool = False):
    f, r = color.value

    def g(x: ndarray) -> ndarray:
        y = cv2.cvtColor(x if gamma is None else x**gamma, f)
        return y.transpose(2, 0, 1) if transpose else y

    def h(x: ndarray) -> ndarray:
        z = cv2.cvtColor(x.transpose(1, 2, 0) if transpose else x, r).clip(0.0, 1.0)
        return z if gamma is None else z ** (1 / gamma)

    return g, h


def split_alpha(x: ndarray) -> tuple[ndarray, ndarray | None]:
    if x.shape[2] == 4:
        return x[:, :, :3], x[:, :, 3]
    elif x.shape[2] == 3:
        return x, None
    raise ValueError()


def merge_alpha(x: ndarray, a: ndarray | None = None) -> ndarray:
    if a is None:
        return x
    return np.concatenate((x, a[:, :, np.newaxis]), axis=2)
