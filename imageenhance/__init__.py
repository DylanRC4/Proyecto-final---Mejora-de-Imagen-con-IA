"""ImageEnhance AI: eliminación de ruido con filtros clásicos y una CNN entrenada desde cero.

Convención interna: imágenes float32 en [0, 1] con forma (H, W, C) en el orden en que
OpenCV las carga (BGR). Solo io_utils convierte a RGB para mostrar en pantalla.
"""

__version__ = "0.1.0"
