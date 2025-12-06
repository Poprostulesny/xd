# extractor.py
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift
from numpy.linalg import eig

# ------------------------------------
# 1. Luminance
# ------------------------------------
def rgb_to_luminance(img):
    return (
        0.2126 * img[:, :, 2] +
        0.7152 * img[:, :, 1] +
        0.0722 * img[:, :, 0]
    ).astype(np.float32)


# ------------------------------------
# 2. Gradient PCA
# ------------------------------------
def gradient_pca_features(lum):
    Gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)

    # NORMALIZACJA ROZMYĆ (bardzo ważne)
    scale = np.mean(np.abs(Gx)) + np.mean(np.abs(Gy)) + 1e-6
    Gx /= scale
    Gy /= scale

    M = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    C = (M.T @ M) / M.shape[0]

    vals, _ = eig(C)
    vals = np.sort(vals)[::-1]
    λ1, λ2 = vals

    rho = λ1 / (λ2 + 1e-9)
    E = λ1 + λ2
    kappa = ((λ1 - λ2) / (λ1 + λ2 + 1e-9))**2

    return rho, E, kappa


# ------------------------------------
# 3. Frequency domain (FFT)
# ------------------------------------
def frequency_features(lum):
    F = fftshift(fft2(lum))
    P = np.abs(F) ** 2

    H, W = lum.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)

    radial = []
    for ri in range(int(r.max())):
        mask = (r == ri)
        if mask.any():
            radial.append(P[mask].mean())
    radial = np.array(radial)

    # slope
    x = np.arange(1, len(radial))
    y = radial[1:]
    slope = -np.polyfit(np.log(x), np.log(y + 1e-9), 1)[0]

    # high freq ratio (top 25%)
    split = int(len(radial) * 0.75)
    hf = radial[split:].sum() / (radial.sum() + 1e-9)

    return slope, hf


# ------------------------------------
# 4. LBP
# ------------------------------------
def lbp_features(lum):
    lbp = local_binary_pattern(lum, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(60), density=True)
    return float(np.max(hist))


# ------------------------------------
# 5. Blur metric
# ------------------------------------
def blur_metric(lum):
    return float(cv2.Laplacian(lum, cv2.CV_64F).var())


# ------------------------------------
# 6. Full feature extractor
# ------------------------------------
def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image {path}")

    lum = rgb_to_luminance(img)

    rho, E, kappa = gradient_pca_features(lum)
    slope, hf = frequency_features(lum)
    lbp = lbp_features(lum)
    blur = blur_metric(lum)

    return np.array([rho, E, kappa, slope, hf, lbp, blur], dtype=np.float32)
