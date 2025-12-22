import cv2
import numpy as np
from numpy import ndarray
from numpy.fft import fftshift, ifftshift


def ellipse(w, h):
    offset = (w + h) / 2 / (w * h)
    y, x = np.ogrid[-h : h + 1, -w : w + 1]
    o = np.uint8((x / w) ** 2 + (y / h) ** 2 - offset <= 1)
    return o


def fft(channel):
    fftimg = cv2.dft(channel, flags=(cv2.DFT_SCALE + cv2.DFT_COMPLEX_OUTPUT))
    return fftshift(fftimg)


def ifft(fftimg):
    img_back = cv2.idft(ifftshift(fftimg))
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


def spectrum_normalized(fftimg):
    def normalization(h, w):
        x = np.arange(w)
        y = np.arange(h)
        cx = x - w // 2
        cy = y - h // 2
        energy = np.sqrt(cx.reshape((1, w)) ** 2 + cy.reshape((h, 1)) ** 2 + 1e-8)
        return energy

    height, width, _ = fftimg.shape
    coefs = normalization(height, width)
    spectrum = 20 * np.log(cv2.magnitude(fftimg[:, :, 0], fftimg[:, :, 1]) * coefs)
    return np.maximum(0, spectrum)


def find_threshold(spectrums: list[ndarray], *, cutoff_rate=0.05):
    n = len(spectrums)
    thresholds: list[int] = []
    for s in spectrums:
        hist = np.bincount(s.ravel().astype(np.uint8))
        hist_alt = hist[1:]
        peak = hist_alt.argmax()
        peak_count = hist_alt[peak]
        window_size = 3
        cut_index = int((sum(hist_alt[peak + i + 1 : -window_size + i] for i in range(window_size)) / window_size < peak_count * cutoff_rate).argmax().item() + peak + 1)
        thresholds.append(cut_index + 1)
    threshold = sum(thresholds) / n
    return threshold


def descreen(x: ndarray, *, auto_threshold: bool = True, threshold: float = 85.0, cmyk: bool = False):
    if cmyk:
        pass

    # 外縁部のアーティファクト対策で余白を追加
    margin: int = 5
    w = np.pad(x, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")
    dest = np.zeros_like(w)
    z = w * 255.0
    _, height, width = z.shape

    # TODO: parameterize
    middle_ratio = 1 / 4
    mid = 1 / middle_ratio * 2
    ew, eh = int(width / mid), int(height / mid)
    pw, ph = (width - ew * 2) // 2, (height - eh * 2) // 2
    middle = np.pad(ellipse(ew, eh), ((ph, height - ph - eh * 2 - 1), (pw, width - pw - ew * 2 - 1)), "constant")

    ffts = [fft(channel) for channel in z]
    spectrums = [spectrum_normalized(f) for f in ffts]
    if auto_threshold:
        threshold = find_threshold(spectrums)
    for i, (f, s) in enumerate(zip(ffts, spectrums)):
        _, thresh = cv2.threshold(s, threshold, 255.0, cv2.THRESH_BINARY)

        radius: int = 2
        # ピーク周辺を広げる（安全マージンを確保）
        # cv2.getStructuringElement は実装に問題があり使わない
        kernel = ellipse(radius, radius)
        thresh = cv2.dilate(thresh, kernel)
        # 中心を保護（dilate の侵入も打ち消す）
        thresh *= 1 - middle
        # なだらかにしてリンギングを減らす
        sigma = radius / 3.0
        ksize = (2 * radius + 1, 2 * radius + 1)
        thresh = cv2.GaussianBlur(thresh, ksize, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

        mask = 1.0 - thresh / 255.0
        img_back = f * mask.reshape((height, width, 1))
        dest[i] = (ifft(img_back) / 255.0).clip(0.0, 1.0)
    result = dest[:, margin:-margin, margin:-margin]

    if cmyk:
        pass

    result = result.clip(0.0, 1.0)
    return result
