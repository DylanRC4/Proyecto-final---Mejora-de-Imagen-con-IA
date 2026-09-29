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


def add_downscale(img: np.ndarray, factor: float) -> np.ndarray:
    """Baja resolución: reduce la foto `factor` veces y la devuelve a su tamaño con interpolación bicúbica.
    El tamaño no cambia, pero el detalle fino se perdió: es lo que la red debe recuperar."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0).astype(np.float32)


def add_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """Sub/sobreexposición física: pasa a luz LINEAL (se deshace la curva sRGB), multiplica la luz por
    2^ev y vuelve a sRGB. −2 EV = la cuarta parte de la luz. Lo que pasa de 1 se satura (se quema)."""
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4) * 2.0 ** ev
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 0.0031308) ** (1 / 2.4) - 0.055).astype(np.float32)


def add_cast(img: np.ndarray, gains=(0.8, 1.0, 1.15)) -> np.ndarray:
    """Dominante de color: ganancias por canal B, G, R (por defecto, luz cálida: menos azul, más rojo)."""
    return np.clip(img * np.asarray(gains, np.float32), 0.0, 1.0)


def random_detail(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degradación aleatoria para entrenar la CNN de DETALLE, en el orden de una cámara real:
    lente (desenfoque) → sensor (baja resolución, algo de ruido) → compresión (JPEG).
    A veces no se aplica nada (≈4 %): así la red también aprende a NO cambiar una foto que ya está bien."""
    x = img
    if rng.random() < 0.7:
        x = add_blur(x, rng.uniform(0.5, 2.5))
    if rng.random() < 0.5:
        x = add_downscale(x, float(rng.choice([1.5, 2.0, 3.0])))
    if rng.random() < 0.3:
        x = add_gaussian(x, rng.uniform(1, 5), seed=int(rng.integers(1 << 31)))
    if rng.random() < 0.6:
        x = add_jpeg(x, int(rng.integers(10, 61)))
    return x


def random_detail_v2(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mezcla EQUILIBRADA (versión 2). La v1 aplicaba desenfoque al 70 % de los ejemplos y la red
    aprendió a "afilar" siempre, empeorando fotos que solo tenían JPEG. Aquí cada caso tiene su cuota:
      10 % foto limpia (aprender a no tocar) | 20 % solo desenfoque | 20 % solo JPEG
      15 % solo baja resolución              | 35 % combinación (random_detail)"""
    u = rng.random()
    if u < 0.10:
        return img.copy()
    if u < 0.30:
        return add_blur(img, rng.uniform(0.5, 2.5))
    if u < 0.50:
        return add_jpeg(img, int(rng.integers(10, 81)))
    if u < 0.65:
        return add_downscale(img, float(rng.choice([1.5, 2.0, 3.0])))
    return random_detail(img, rng)


VAL_DETAIL = [lambda x: add_jpeg(add_downscale(add_blur(x, 1.0), 2.0), 35), lambda x: add_blur(x, 1.5),
              lambda x: add_jpeg(x, 20), lambda x: add_downscale(x, 2.0)]


def fixed_detail(img: np.ndarray) -> np.ndarray:
    """Degradación FIJA de validación y prueba: desenfoque 1.0 → reducción ×2 → JPEG calidad 35."""
    return add_jpeg(add_downscale(add_blur(img, 1.0), 2.0), 35)


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
    if kind == "downscale":
        return add_downscale(img, level)
    raise ValueError(f"Tipo de degradación desconocido: {kind}")
