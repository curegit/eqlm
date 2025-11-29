from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtCore import QBuffer, QByteArray

app = QGuiApplication.instance() or QGuiApplication()


def import_as_png_from_clipboard() -> bytes:
    clipboard = QGuiApplication.clipboard()
    mime_data = clipboard.mimeData()
    image = None
    if mime_data.hasImage():
        image = mime_data.imageData()
    if image:
        assert isinstance(image, QImage)
    else:
        raise OSError("No image data in the clipboard, or an unsupported image type")
    buf = QBuffer()
    ok = image.save(buf, "PNG", 100)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    bits: QByteArray = buf.data()
    return bytes(bits.data())


def save_to_clipboard(png_bytes: bytes):
    clipboard = QGuiApplication.clipboard()
    # QImage::Format_RGB32 / QImage::Format_ARGB32 (this preserves ICC profiles)
    image = QImage.fromData(png_bytes)
    clipboard.setImage(image)
