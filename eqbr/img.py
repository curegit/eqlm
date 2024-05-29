import io
import struct
import zlib
import cv2
import numpy as np
from pathlib import Path
from enum import Enum
from io import BufferedIOBase
from numpy import ndarray
from PIL import Image


def load_image(filelike: str | Path | bytes | memoryview, *, normalize: bool = True) -> tuple[ndarray, bytes | None]:
    match filelike:
        case str() | Path() as path:
            with open(Path(path).resolve(strict=True), "rb") as fp:
                buffer = fp.read()
        case bytes() | memoryview() as buffer:
            pass
        case _:
            raise ValueError()
    icc = extract_icc(buffer)
    # OpenCV が ASCII パスしか扱えない問題を回避するためにバッファを経由する
    bin = np.frombuffer(buffer, np.uint8)
    # flags = cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH
    # if not orient:
    #    flags |= cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(bin, cv2.IMREAD_UNCHANGED)
    match img.dtype:
        case np.uint8:
            if normalize:
                return (img / (2**8 - 1)).astype(np.float32), icc
            else:
                return img, icc
        case np.uint16:
            if normalize:
                return (img / (2**16 - 1)).astype(np.float32), icc
            else:
                return img, icc
        case np.float32:
            return img, icc
        case _:
            raise RuntimeError()


def save_image(img: ndarray, filelike: str | Path | BufferedIOBase, *, prefer16=False, icc_profile: bytes|None=None) -> None:
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
    if icc_profile is not None:
        print("ICC")
        buffer = embed_icc_png(buffer, icc_profile)


    match filelike:
        case str() | Path() as path:
            with open(Path(path), "wb") as fp:
                fp.write(buffer)
        case BufferedIOBase() as stream:
            stream.write(buffer)
        case _:
            raise ValueError()


def extract_icc(img_bytes: bytes | memoryview) -> bytes | None:
    buf = io.BytesIO(img_bytes)
    image = Image.open(buf)
    maybe_icc = image.info.get("icc_profile")
    if maybe_icc is None:
      return None
    else:
      assert isinstance(maybe_icc, bytes)
      return maybe_icc



def embed_icc_png(png_bytes: bytes, icc_profile: bytes) -> bytes:
    assert png_bytes[:8] == bytes.fromhex("89504E470D0A1A0A")

    chunk_type = None
    offset = 8
    while chunk_type != b'IDAT':
        (length,) = struct.unpack("!I", png_bytes[offset: offset+4])
        chunk_type = png_bytes[offset+4: offset+8]
        assert chunk_type != b"sRGB" and chunk_type != b"iCCP"
        assert ((offset == 8 and length == 13) if chunk_type == b"IHDR" else True)
        offset += 4 + 4 + length + 4


    compobj = zlib.compressobj(method=zlib.DEFLATED)
    deflated = compobj.compress(icc_profile)
    deflated += compobj.flush()
    iccp_chunk_type = b'iCCP'
    iccp_chunk_data = b'ICC Profile' + bytes.fromhex("0000") + deflated
    iccp_length = struct.pack("!I", len(iccp_chunk_data))
    iccp_crc = struct.pack("!I", zlib.crc32(iccp_chunk_type + iccp_chunk_data, 0))
    iccp_chunk = iccp_length + iccp_chunk_type + iccp_chunk_data + iccp_crc
    return png_bytes[:33] + iccp_chunk + png_bytes[33:]


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
