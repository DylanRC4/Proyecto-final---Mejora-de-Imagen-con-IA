"""Degradaciones sintéticas reproducibles.

Taller: Sesión 04. Generamos el ruido en vez de descargarlo para controlar su proporción
exacta y conservar la imagen limpia como referencia. Es el mismo principio que usan los
pares (limpia, ruidosa) con los que se entrena la CNN de forma supervisada.
Todas las funciones reciben float32 en [0, 1] y una semilla explícita.
"""
import cv2
import numpy as np


def add_gaussian(img: np.ndarray, sigma: float, seed: int | None = None, clip: bool = True) -> np.ndarray:
    """Ruido gaussiano aditivo N(0, sigma²) con sigma en la escala 0-255 (sigma=25 es el valor clásico).

    El ruido afecta a todos los píxeles. Con clip=True se simula que la foto ruidosa se guardó
    como imagen normal (valores dentro de [0, 1]).
    """
    rng = np.random.default_rng(seed)
    noisy = img + rng.normal(0.0, sigma / 255.0, img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0) if clip else noisy


def add_salt_pepper(img: np.ndarray, amount: float = 0.05, seed: int | None = None) -> np.ndarray:
    """Ruido impulsivo: una fracción `amount` de los píxeles pasa a negro o blanco (mitad y mitad).

    Se afecta el píxel completo (todos sus canales), como en el laboratorio 06 (5 %, semilla 42).
    """
    rng = np.random.default_rng(seed)
    out = img.copy()  # sin copy dañaríamos la original (Sesión 04)
    u = rng.random(img.shape[:2])
    out[u < amount / 2] = 0.0
    out[(u >= amount / 2) & (u < amount)] = 1.0
    return out


def add_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque: convolución con un kernel gaussiano (Sesión 04), como una lente mal enfocada."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma)


def add_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Compresión JPEG real con OpenCV: a menor calidad, más bloques de 8x8 visibles."""
    u8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1]
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def degrade(img: np.ndarray, kind: str = "gaussian", level: float = 25, seed: int | None = None) -> np.ndarray:
    """Punto de entrada único: gaussian (level=sigma), salt_pepper (fracción), blur (sigma), jpeg (calidad)."""
    if kind == "gaussian":
        return add_gaussian(img, level, seed)
    if kind == "salt_pepper":
        return add_salt_pepper(img, level, seed)
    if kind == "blur":
        return add_blur(img, level)
    if kind == "jpeg":
        return add_jpeg(img, int(level))
    raise ValueError(f"Tipo de degradación desconocido: {kind}")
