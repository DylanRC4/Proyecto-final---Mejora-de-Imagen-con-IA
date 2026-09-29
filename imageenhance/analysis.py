"""Análisis de una imagen SIN referencia limpia (fotos reales).

- Histogramas y estadísticas por canal (Sesión 02).
- Estimación del nivel de ruido (Immerkær, 1996): convoluciona la imagen en grises con un kernel
  que anula zonas planas y rampas, y deja pasar el ruido. Es una ESTIMACIÓN (las texturas finas la
  inflan); no reemplaza al PSNR, que necesita la imagen limpia.
- Magnitud del gradiente con Sobel (Sesión 05) para ver si un filtro conserva los bordes.
"""
import cv2
import numpy as np

from .classic import conv2d
from .io_utils import to_gray

IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def channel_stats(img: np.ndarray) -> dict:
    """Media, desviación, mínimo y máximo por canal B, G, R en escala 0-255."""
    x = img.astype(np.float32) * (255.0 if img.dtype != np.uint8 else 1.0)
    return {c: {"media": float(x[..., i].mean()), "desv": float(x[..., i].std()),
                "min": float(x[..., i].min()), "max": float(x[..., i].max())} for i, c in enumerate("BGR")}


def histograms(img: np.ndarray, bins: int = 256) -> dict:
    u8 = img if img.dtype == np.uint8 else np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return {c: np.bincount(u8[..., i].ravel(), minlength=bins) for i, c in enumerate("BGR")}


def estimate_noise_sigma(img_bgr: np.ndarray) -> float:
    """sigma ≈ sqrt(pi/2) · mean(|I * M|) / 6 por canal, promediado; escala 0-255 (Immerkær, 1996).

    Se estima por canal y no sobre la imagen en grises: al ponderar los canales (Sesión 02) el
    ruido independiente de cada uno se promedia y el sigma medido en grises sale ~33 % menor.
    """
    x = img_bgr.astype(np.float32) * (255.0 if img_bgr.dtype != np.uint8 else 1.0)
    r = conv2d(x, IMMERKAER)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)


def estimate_flat_noise(img_bgr: np.ndarray, block: int = 8, q: float = 10) -> float:
    """Ruido en las zonas PLANAS (escala 0-255): a cada bloque 8x8 se le quita su plano (brillo y
    rampa) y se mide la desviación que queda; se toma el percentil 10, que corresponde a los bloques
    más lisos (cielo, paredes). A diferencia de Immerkær, también ve el grano "en manchitas" del
    celular: en una prueba subió de 1.7 a 5.5 con ruido de cámara, mientras Immerkær casi no cambió."""
    g = to_gray(img_bgr).astype(np.float64) * 255
    h, w = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    blocks = g[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    yy, xx = np.mgrid[0:block, 0:block]
    X = np.c_[np.ones(block * block), yy.ravel(), xx.ravel()]
    coef = np.linalg.lstsq(X, blocks.T, rcond=None)[0]
    return float(np.percentile((blocks - (X @ coef).T).std(axis=1), q))


def gradient_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    """G = sqrt(Gx² + Gy²) con Sobel 3x3 sobre la luminancia, en float64 (la derivada sale de 0-255)."""
    g = to_gray(img_bgr).astype(np.float64)
    gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)
