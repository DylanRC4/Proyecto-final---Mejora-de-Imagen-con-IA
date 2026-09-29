"""Mejoras clásicas de iluminación, contraste, color y nitidez (sin entrenamiento).

Son problemas GLOBALES de la foto (toda la imagen demasiado oscura, apagada o con un tono de
color), y se corrigen bien midiendo el histograma. Todas reciben y devuelven float32 BGR en [0, 1].
Talleres: Sesión 01 (operaciones puntuales), 02 (luminancia e histogramas), 04 (filtro gaussiano).
"""
import cv2
import numpy as np

from .classic import gaussian_filter
from .io_utils import resize_max, to_gray, to_uint8


def auto_gamma(img: np.ndarray, target: float = 0.45) -> tuple[np.ndarray, float]:
    """Iluminación: A' = A^γ, con γ elegido para que la luminancia media quede en `target`.

    γ < 1 aclara (foto oscura) y γ > 1 oscurece (foto quemada). A diferencia de la transformación
    afín (Sesión 01), es no lineal: levanta las sombras sin quemar las zonas claras.
    """
    mean = float(np.clip(to_gray(img).mean(), 0.02, 0.98))
    gamma = float(np.clip(np.log(target) / np.log(mean), 0.4, 2.5))
    return np.power(img, gamma, dtype=np.float32), gamma


def auto_levels(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[np.ndarray, float, float]:
    """Contraste global: transformación afín A' = α·A + β (Sesión 01) con α y β calculados del
    histograma, para que el percentil `low` de la luminancia pase a 0 y el `high` a 1.
    Se usan percentiles y no el mínimo/máximo para que unos pocos píxeles extremos no la anulen."""
    lo, hi = np.percentile(to_gray(img), [low, high])
    alpha = 1.0 / max(float(hi - lo), 0.05)
    beta = -alpha * float(lo)
    return np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32), alpha, beta


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contraste: ecualiza el histograma por zonas (CLAHE) solo en la luminosidad L del espacio LAB.

    Trabajar sobre L y no sobre B, G y R por separado evita cambiar los colores.
    `clip` limita cuánto se estira cada histograma local para no amplificar el ruido.
    """
    lab = cv2.cvtColor(to_uint8(img), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0


def gray_edge(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Hipótesis "gray-edge" (van de Weijer, Gevers y Gijsenij, 2007): en una escena con luz neutra,
    los BORDES promedian gris. Devuelve por canal B, G, R la norma p de la magnitud del gradiente
    Sobel (Sesión 05). Una dominante multiplica cada canal por una ganancia, y sus bordes por la misma.
    A diferencia del "mundo gris" (promedio de colores), un bosque verde o un atardecer no la engañan
    tanto: en validación separa fotos limpias de fotos con dominante con AUC 0.96 (mundo gris: 0.65)."""
    s = resize_max(img, 512).astype(np.float32)
    mag = [np.hypot(cv2.Sobel(s[..., c], cv2.CV_32F, 1, 0), cv2.Sobel(s[..., c], cv2.CV_32F, 0, 1)) for c in range(3)]
    return np.array([np.mean(m.astype(np.float64) ** p) ** (1 / p) for m in mag])


def white_balance(img: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Color: balance de blancos gray-edge. Escala cada canal para que la "energía" de sus bordes
    iguale a la de los otros. Ganancias limitadas a [0.7, 1.4]. En validación, una dominante del
    10-25 % pasa de 24.3 dB a 34.0 dB, y si se aplica por error a una foto limpia la deja en 39 dB."""
    v = gray_edge(img)
    gains = np.clip(v.mean() / np.maximum(v, 1e-6), 0.7, 1.4)
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(img * gains, 0.0, 1.0).astype(np.float32), gains


def unsharp(img: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    """Nitidez: máscara de desenfoque. Resta a la imagen su versión suavizada (gaussiano, Sesión 04)
    para aislar los bordes y los suma de nuevo amplificados:  A' = A + amount·(A − G∗A)."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return np.clip(img + amount * (img - gaussian_filter(img, k, sigma)), 0.0, 1.0).astype(np.float32)
