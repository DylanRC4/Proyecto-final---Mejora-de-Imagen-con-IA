"""Diagnóstico automático: ¿qué le pasa a esta foto?

1. Se extraen 11 características numéricas de la imagen (Sesión 06: de píxeles a una tabla).
2. Ruido, desenfoque y compresión JPEG los decide un clasificador MLP entrenado por nosotros
   (Sesión 12), porque ahí las medidas se confunden entre sí: la textura parece ruido, el
   desenfoque esconde el ruido, la compresión borra detalle...
3. Iluminación, contraste y color se deciden con reglas sobre el histograma (Sesión 02): son
   problemas globales y un umbral claro basta.
"""
import cv2
import numpy as np

from .analysis import estimate_noise_sigma
from .io_utils import resize_max, to_gray

FEATURES = ["ruido_sigma", "nitidez_log_var_laplaciano", "alta_frec_relativa", "densidad_bordes_canny",
            "bloques_jpeg", "brillo", "contraste", "percentil_1", "percentil_99",
            "dominante_media", "dominante_brillantes"]
LABELS = ["ruido", "desenfoque", "compresion"]
CROP = 768  # recorte central para las medidas de detalle (BSDS500 mide 481 px: cabe completa)


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
    """
    y, x = (max(0, (d - CROP) // 2) for d in img.shape[:2])
    crop = img[y:y + CROP, x:x + CROP]
    g = to_gray(crop).astype(np.float64) * 255  # float64: la derivada sale del rango (Sesión 05)
    lap = cv2.Laplacian(g, cv2.CV_64F)  # segunda derivada (Sesión 05): responde a bordes finos y ruido
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    edges = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), 50, 150)  # umbrales de la Sesión 05
    small = resize_max(img, 512)
    gs = to_gray(small)
    means = small.reshape(-1, 3).mean(axis=0)
    bright = small[gs >= np.percentile(gs, 95)].mean(axis=0)  # los píxeles más claros suelen ser blancos o grises
    p1, p99 = np.percentile(gs, [1, 99])
    return np.array([estimate_noise_sigma(crop), np.log10(lap.var() + 1), np.abs(lap).mean() / (grad.mean() + 1e-6),
                     (edges > 0).mean(), blockiness(g), gs.mean(), gs.std(), p1, p99,
                     np.ptp(means) / (means.mean() + 1e-6), np.ptp(bright) / (bright.mean() + 1e-6)], dtype=np.float32)


def tone_flags(f: np.ndarray) -> dict:
    """Reglas de iluminación, contraste y color. Umbrales conservadores, medidos sobre las 100 fotos
    de validación: solo actúan en casos claros. El color es el caso más difícil: sin referencia no
    se distingue un atardecer de una foto amarillenta, por eso exige dos medidas a la vez."""
    brillo, p1, p99, dom_media, dom_brillantes = f[5], f[7], f[8], f[9], f[10]
    return {"oscura": bool(brillo < 0.25), "sobreexpuesta": bool(brillo > 0.72),
            "poco_contraste": bool(p99 - p1 < 0.45),
            "dominante_color": bool(dom_media > 0.35 and dom_brillantes > 0.35)}


def build_mlp(n_in: int = len(FEATURES), n_out: int = len(LABELS)):
    """11 → 32 → 16 → 3. Parámetros: (11+1)·32 + (32+1)·16 + (16+1)·3 = 963 (fórmula de la Sesión 12)."""
    from torch import nn
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_out))


class Diagnoser:
    """Carga el MLP entrenado (models/diagnosis_mlp.pt) y diagnostica una foto."""

    def __init__(self, path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
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
