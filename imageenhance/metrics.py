"""Métricas de calidad CON referencia: PSNR y SSIM.

Solo son válidas cuando existe la imagen limpia original (ruido agregado por nosotros).
En una foto real con ruido no existe esa referencia y estas métricas no se pueden calcular;
para ese caso analysis.py ofrece una ESTIMACIÓN sin referencia del nivel de ruido.

Taller: Sesión 04 mostró que el error medio contra la original no bastaba para comparar
filtros. PSNR resume el error cuadrático; SSIM compara estructura local (media, varianza y
covarianza en ventanas gaussianas, es decir, con convoluciones).
"""
import cv2
import numpy as np

_C1, _C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2  # constantes de Wang et al. (2004) para rango 1


def mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))


def psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR = 10·log10(R² / MSE) en dB. Más alto es mejor; +inf si las imágenes son idénticas."""
    m = mse(ref, test)
    return float("inf") if m == 0 else float(10 * np.log10(data_range ** 2 / m))


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    s = ((2 * mx * my + _C1) * (2 * cxy + _C2)) / ((mx ** 2 + my ** 2 + _C1) * (vx + vy + _C2))
    return float(s[5:-5, 5:-5].mean())  # se descarta el borde afectado por el relleno


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """SSIM (Wang et al., 2004) con ventana gaussiana 11x11, sigma 1.5; promedio de canales. Rango (-1, 1]."""
    x, y = ref.astype(np.float64), test.astype(np.float64)
    if x.ndim == 2:
        return _ssim_channel(x, y)
    return float(np.mean([_ssim_channel(x[..., c], y[..., c]) for c in range(x.shape[-1])]))


def evaluate(ref: np.ndarray, test: np.ndarray) -> dict:
    return {"psnr": psnr(ref, test), "ssim": ssim(ref, test)}
