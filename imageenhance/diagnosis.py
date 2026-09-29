"""Diagnóstico automático: ¿qué le pasa a esta foto?

1. Se extraen 11 características numéricas de la imagen (Sesión 06: de píxeles a una tabla).
2. Ruido, desenfoque y compresión JPEG los decide un clasificador MLP entrenado por nosotros
   (Sesión 12), porque ahí las medidas se confunden entre sí: la textura parece ruido, el
   desenfoque esconde el ruido, la compresión borra detalle...
3. Iluminación y contraste se deciden con reglas sobre el histograma (Sesión 02) y el color con la
   hipótesis gray-edge: son problemas globales y un umbral claro, elegido en validación, basta.
"""
import cv2
import numpy as np

from .analysis import estimate_flat_noise, estimate_noise_sigma
from .enhance import gray_edge
from .io_utils import resize_max, to_gray

FEATURES = ["ruido_sigma", "nitidez_log_var_laplaciano", "alta_frec_relativa", "densidad_bordes_canny",
            "bloques_jpeg", "brillo", "contraste", "percentil_1", "percentil_99",
            "dominante_bordes", "ruido_zonas_planas"]
LABELS = ["ruido", "desenfoque", "compresion"]
CROP = 768  # recorte central para las medidas de detalle (BSDS500 mide 481 px: cabe completa)
# Umbral de dominante de color elegido en VALIDACIÓN (100 fotos limpias + 100 con dominante aleatoria del
# 10-25 %): el menor con falsas alarmas <= 5 %. Con 0.18: 5 % de falsas alarmas y detecta el 87 %.
# La regla anterior (mundo gris) tenía 15 % de falsas alarmas y detectaba solo el 40 %.
# OJO: en fotos modernas (DIV2K val) este umbral da 33 % de falsas alarmas: por eso el color solo se avisa.
COLOR_UMBRAL = 0.18
# Tono: se juzga el RANGO del histograma, no el brillo medio. Subexponer escala toda la foto hacia abajo, así
# que su percentil 99 no llega al blanco; una escena nocturna bien expuesta sí tiene luces que llegan. Umbrales
# elegidos en VALIDACIÓN (100 fotos): con p99 < 0.6 o p1 > 0.25 o rango < 0.45 se detecta el 100 % de las fotos
# a −2 EV (antes 86 %) con 8 falsas alarmas de 100 (antes 7); el PSNR tras corregir sube de 20.96 a 23.00 dB
# a −2 EV y de 18.66 a 24.28 dB con neblina. La gamma solo actúa si, tras los niveles, el brillo sigue fuera
# de [BRILLO_MIN, BRILLO_MAX].
P99_MIN, P1_MAX, RANGO_MIN, BRILLO_MIN, BRILLO_MAX = 0.6, 0.25, 0.45, 0.25, 0.72


def blockiness(gray: np.ndarray) -> float:
    """Cuánto más fuertes son los saltos entre píxeles en las fronteras de los bloques 8x8 del JPEG
    que en el resto de la imagen. Foto natural ≈ 1; JPEG de baja calidad > 1."""
    dh, dv = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    return float((dh[:, 7::8].mean() / (dh.mean() + 1e-6) + dv[7::8].mean() / (dv.mean() + 1e-6)) / 2)


def features(img: np.ndarray) -> np.ndarray:
    """Vector de 11 características de una imagen BGR float [0, 1] a la resolución en que se va a procesar.

    Ruido, nitidez, bordes y bloques JPEG se miden en un recorte central SIN reducir la foto:
    reducirla promedia píxeles vecinos y borra justo el ruido y los bloques que buscamos.
    Brillo, contraste y color son globales, así que se miden sobre la foto reducida a 512 px.
    El color se mide con los bordes (gray-edge): cuánto difiere la energía de los bordes entre canales.
    """
    y, x = (max(0, (d - CROP) // 2) for d in img.shape[:2])
    crop = img[y:y + CROP, x:x + CROP]
    g = to_gray(crop).astype(np.float64) * 255  # float64: la derivada sale del rango (Sesión 05)
    lap = cv2.Laplacian(g, cv2.CV_64F)  # segunda derivada (Sesión 05): responde a bordes finos y ruido
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    edges = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), 50, 150)  # umbrales de la Sesión 05
    small = resize_max(img, 512)
    gs = to_gray(small)
    edges_bgr = gray_edge(small)
    p1, p99 = np.percentile(gs, [1, 99])
    return np.array([estimate_noise_sigma(crop), np.log10(lap.var() + 1), np.abs(lap).mean() / (grad.mean() + 1e-6),
                     (edges > 0).mean(), blockiness(g), gs.mean(), gs.std(), p1, p99,
                     np.ptp(edges_bgr) / (edges_bgr.mean() + 1e-6), estimate_flat_noise(crop)], dtype=np.float32)


def tone_flags(f: np.ndarray) -> dict:
    """Reglas de iluminación, contraste y color, con umbrales elegidos en validación (ver arriba)."""
    p1, p99, dom_bordes = f[7], f[8], f[9]
    return {"oscura": bool(p99 < P99_MIN), "sobreexpuesta": bool(p1 > P1_MAX),
            "poco_contraste": bool(p99 - p1 < RANGO_MIN), "dominante_color": bool(dom_bordes > COLOR_UMBRAL)}


def build_mlp(n_in: int = len(FEATURES), n_out: int = len(LABELS)):
    """11 → 32 → 16 → 3. Parámetros: (11+1)·32 + (32+1)·16 + (16+1)·3 = 963 (fórmula de la Sesión 12)."""
    from torch import nn
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_out))


class Diagnoser:
    """Carga el MLP entrenado (models/diagnosis_mlp.pt) y diagnostica una foto."""

    def __init__(self, path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if list(ck["features"]) != FEATURES:
            raise ValueError("El MLP se entrenó con otras características. Ejecuta: python scripts/train_diagnosis.py")
        self.mean, self.std = ck["mean"].numpy(), ck["std"].numpy()
        self.net = build_mlp()
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

    def __call__(self, img: np.ndarray) -> dict:
        import torch
        f = features(img)
        z = (f - self.mean) / self.std  # misma normalización que en el entrenamiento (Sesión 09)
        with torch.no_grad():
            prob = torch.sigmoid(self.net(torch.from_numpy(z)[None]))[0].numpy()
        return {"features": dict(zip(FEATURES, f.round(4).tolist())),
                "prob": dict(zip(LABELS, prob.round(3).tolist())),
                "problemas": {**{l: bool(p > 0.5) for l, p in zip(LABELS, prob)}, **tone_flags(f)}}
