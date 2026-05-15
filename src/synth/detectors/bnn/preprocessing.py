"""BNN preprocessing — forensic channel computation.

Computes additional image channels used by the BNext backbone
for deepfake detection:

* **Sobel magnitude** — edge gradient strength (detects GAN boundary artifacts)
* **FFT magnitude** — frequency-domain view (detects spectral artifacts)
* **LBP** — Local Binary Pattern (detects micro-texture anomalies)

These are concatenated with the RGB channels and fed through a
learned 6→3 adapter convolution.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude from a grayscale image.

    Args:
        gray: HxW uint8 grayscale image.

    Returns:
        HxW float64 magnitude image, normalised to [0, 1].
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalise to [0, 1]
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max
    return magnitude


def compute_fft_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute FFT log-magnitude spectrum from a grayscale image.

    Args:
        gray: HxW uint8 grayscale image.

    Returns:
        HxW float64 log-magnitude spectrum, normalised to [0, 1].
    """
    fft = np.fft.fft2(gray.astype(np.float64))
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20.0 * np.log(np.abs(fft_shift) + 1e-9)
    # Normalise to [0, 1]
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    if mag_max > mag_min:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)
    return magnitude


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    """Compute Local Binary Pattern from a grayscale image.

    Uses a simple 8-neighbour LBP implementation (no scikit-image dependency).

    Args:
        gray: HxW uint8 grayscale image.

    Returns:
        HxW float64 LBP image, normalised to [0, 1].
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # Pad the image
    padded = np.pad(gray, 1, mode="edge")

    # 8-neighbour offsets (clockwise from top-left)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1),
    ]

    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[1 + dy: h + 1 + dy, 1 + dx: w + 1 + dx]
        lbp |= ((neighbour >= padded[1: h + 1, 1: w + 1]).astype(np.uint8) << bit)

    return lbp.astype(np.float64) / 255.0


def compute_forensic_channels(
    image_rgb: np.ndarray,
) -> np.ndarray:
    """Compute all 3 forensic channels from an RGB image.

    Args:
        image_rgb: HxWx3 uint8 RGB image.

    Returns:
        HxWx3 float64 array of [sobel, fft, lbp] channels.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    sobel = compute_sobel_magnitude(gray)
    fft = compute_fft_magnitude(gray)
    lbp = compute_lbp(gray)

    return np.stack([sobel, fft, lbp], axis=-1)
