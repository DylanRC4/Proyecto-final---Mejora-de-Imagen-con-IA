"""Procesamiento tradicional: filtros fijos diseñados a mano (línea base frente a la CNN).

Talleres:
- Sesión 01: kernel de convolución programado con NumPy (Hadamard + suma) y transformación
  afín A' = alfa*A + beta para contraste y brillo.
- Sesión 04: filtros de suavizado. La mediana funciona con ruido impulsivo; el gaussiano con
  ruido que afecta a todos los píxeles. La mediana NO es una convolución (no es lineal).
Todas las funciones reciben y devuelven float32 en [0, 1] con forma (H, W) o (H, W, C).
"""
import cv2
import numpy as np

# Laplaciano discreto: identidad + este kernel da el kernel de realce de la Sesión 01 (5 al centro).
LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución 2D propia en NumPy, canal por canal, con el mismo borde que OpenCV (reflect-101).

    Igual que en OpenCV y en PyTorch (nn.Conv2d), es una correlación: cada salida es la suma del
    producto posición contra posición entre el kernel y la vecindad. En vez de recorrer píxel por
    píxel, se recorre el kernel: kh*kw desplazamientos vectorizados de toda la imagen.
    """
    k = np.asarray(kernel, dtype=np.float32)
    kh, kw = k.shape
    x = img.astype(np.float32)
    h, w = x.shape[:2]
    pad = ((kh // 2, kh // 2), (kw // 2, kw // 2)) + ((0, 0),) * (x.ndim - 2)
    xp = np.pad(x, pad, mode="reflect")  # 'reflect' de NumPy = BORDER_REFLECT_101 de OpenCV
    out = np.zeros_like(x)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * xp[i:i + h, j:j + w]
    return out


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Kernel gaussiano 2D normalizado (suma 1: una zona plana no cambia de brillo)."""
    r = np.arange(size, dtype=np.float32) - (size - 1) / 2
    g = np.exp(-(r ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def box_kernel(size: int) -> np.ndarray:
    """Filtro de media: todos los vecinos pesan igual (1/size²)."""
    return np.full((size, size), 1.0 / size ** 2, dtype=np.float32)


def gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 1.0, own: bool = False) -> np.ndarray:
    """Suavizado gaussiano. own=True usa nuestra conv2d; own=False usa OpenCV (mismo resultado, más rápido)."""
    if own:
        return conv2d(img, gaussian_kernel(ksize, sigma))
    return cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigma)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Mediana: ordena la vecindad y toma el valor central; los extremos 0/255 aislados nunca quedan en el medio.

    OpenCV solo admite float32 con ksize 3 o 5; para kernels mayores se trabaja en uint8.
    """
    if ksize <= 5:
        return cv2.medianBlur(img.astype(np.float32), ksize)
    u8 = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    return cv2.medianBlur(u8, ksize).astype(np.float32) / 255.0


def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Transformación afín A' = alfa*A + beta (Sesión 01). alfa = contraste, beta = brillo en escala 0-255."""
    return np.clip(alpha * img + beta / 255.0, 0.0, 1.0)


def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Realce de nitidez con el kernel identidad + amount*Laplaciano (amount=1 es el kernel de la Sesión 01).

    Los coeficientes suman 1, así que las zonas planas no cambian; solo actúa donde hay contraste.
    Ojo: también amplifica el ruido, por eso se aplica DESPUÉS de eliminarlo.
    """
    k = amount * LAPLACIAN
    k[1, 1] += 1.0
    return np.clip(conv2d(img, k), 0.0, 1.0)


def apply(img: np.ndarray, method: str, **params) -> np.ndarray:
    """Despachador usado por los scripts y la interfaz."""
    fns = {"median": median_filter, "gaussian": gaussian_filter,
           "mean": lambda x, ksize=3: conv2d(x, box_kernel(ksize))}
    if method not in fns:
        raise ValueError(f"Método desconocido: {method}")
    return fns[method](img, **params)
