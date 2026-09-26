"""Entrada/salida y conversiones de tipo y color.

Talleres: Sesión 01 (una imagen es una matriz; calcular en float32 y hacer clip antes
de volver a uint8) y Sesión 02 (OpenCV carga en BGR; pesos de luminancia en orden BGR).
"""
from pathlib import Path

import cv2
import numpy as np

# Pesos de luminancia en orden BGR (Sesión 02): el ojo es más sensible al verde.
LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def decode_bytes(data: bytes) -> np.ndarray:
    """Decodifica bytes de un archivo de imagen (p. ej. subido en Streamlit) a BGR uint8."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return img


def load_bgr(path) -> np.ndarray:
    """Carga una imagen a color como BGR uint8. Usa imdecode para admitir rutas con tildes en Windows."""
    return decode_bytes(Path(path).read_bytes())


def save_bgr(path, img: np.ndarray) -> None:
    """Guarda una imagen BGR (uint8 o float en [0, 1]) respetando rutas con tildes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", to_uint8(img) if img.dtype != np.uint8 else img)
    if not ok:
        raise ValueError(f"No se pudo codificar {path}")
    path.write_bytes(buf.tobytes())


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. Se calcula en float para no perder decimales (Sesión 01)."""
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8. Clip antes de convertir: uint8 es aritmética modular (-10 daría 246)."""
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Invierte el eje de canales (BGR <-> RGB). Streamlit y Matplotlib esperan RGB."""
    return np.ascontiguousarray(img[..., ::-1])


rgb_to_bgr = bgr_to_rgb


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Escala de grises ponderada (Sesión 02): producto punto de cada píxel con los pesos BGR."""
    return (img_bgr.astype(np.float32) @ LUMA_BGR).astype(np.float32)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """Reduce la imagen para que su lado mayor no pase de max_side (INTER_AREA evita aliasing)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
