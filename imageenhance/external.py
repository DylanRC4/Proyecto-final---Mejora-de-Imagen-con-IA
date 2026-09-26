"""Modelos EXTERNOS preentrenados (no entrenados por nosotros). Se identifican siempre en la app.

Real-ESRGAN, versión compacta "realesr-general-x4v3" (Wang et al., 2021, licencia BSD-3):
superresolución ×4 que además limpia desenfoque leve, ruido y compresión, porque fue entrenada
con esas degradaciones combinadas. La arquitectura (SRVGGNetCompact) la escribimos aquí para
poder explicarla; los pesos se descargan con scripts/download_models.py.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EXTERNAL = Path(__file__).resolve().parents[1] / "models" / "external"
ESRGAN_PATH = EXTERNAL / "realesr-general-x4v3.pth"


class SRVGGNetCompact(nn.Module):
    """1 conv de entrada + 32 conv 3x3 de 64 filtros con PReLU + 1 conv de salida con 3·4·4 canales.

    PixelShuffle reordena esos 48 canales en una imagen 4 veces más grande (cada píxel de entrada
    produce un bloque 4x4). Igual que nuestra CNN, es residual: la red solo aprende lo que le falta
    a la ampliación simple por vecino más cercano. Parámetros: 1.213.296.
    """

    def __init__(self, feat: int = 64, n_conv: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        layers = [nn.Conv2d(3, feat, 3, 1, 1), nn.PReLU(feat)]
        for _ in range(n_conv):
            layers += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
        layers.append(nn.Conv2d(feat, 3 * scale * scale, 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.up = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.up(self.body(x)) + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def load_esrgan(path=ESRGAN_PATH) -> SRVGGNetCompact:
    if not Path(path).exists():
        raise FileNotFoundError("Falta el modelo externo. Ejecuta: python scripts/download_models.py")
    net = SRVGGNetCompact()
    net.body.load_state_dict({k.split(".", 1)[1]: v for k, v in
                              torch.load(path, map_location="cpu", weights_only=True)["params"].items()})
    return net.eval()


@torch.no_grad()
def upscale4(net: nn.Module, img_bgr: np.ndarray, tile: int = 192, pad: int = 12) -> np.ndarray:
    """×4 por bloques (para no agotar la RAM del portátil). El modelo trabaja en RGB: convertimos (Sesión 02)."""
    rgb = np.ascontiguousarray(img_bgr[..., ::-1], dtype=np.float32)
    h, w = rgb.shape[:2]
    s = net.scale
    out = np.empty((h * s, w * s, 3), np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0, y1, x1 = max(y - pad, 0), max(x - pad, 0), min(y + tile + pad, h), min(x + tile + pad, w)
            t = torch.from_numpy(rgb[y0:y1, x0:x1].transpose(2, 0, 1).copy())[None]
            r = net(t)[0].numpy().transpose(1, 2, 0)
            th, tw = min(tile, h - y), min(tile, w - x)
            out[y * s:(y + th) * s, x * s:(x + tw) * s] = r[(y - y0) * s:(y - y0 + th) * s, (x - x0) * s:(x - x0 + tw) * s]
    return np.clip(out[..., ::-1], 0.0, 1.0)


def upscale(net: nn.Module, img_bgr: np.ndarray, factor: int = 2) -> np.ndarray:
    """Amplía ×2 o ×4. Para ×2 se amplía ×4 y se reduce a la mitad con INTER_AREA (como hace Real-ESRGAN)."""
    big = upscale4(net, img_bgr)
    h, w = img_bgr.shape[:2]
    return big if factor == 4 else cv2.resize(big, (w * factor, h * factor), interpolation=cv2.INTER_AREA)


def restore(net: nn.Module, img_bgr: np.ndarray) -> np.ndarray:
    """Restaura nitidez y compresión SIN cambiar el tamaño: reduce a la mitad, amplía ×4 y vuelve al
    tamaño original. Trabajar a la mitad hace el proceso 4 veces más rápido en CPU."""
    h, w = img_bgr.shape[:2]
    half = cv2.resize(img_bgr, (max(w // 2, 8), max(h // 2, 8)), interpolation=cv2.INTER_AREA)
    return cv2.resize(upscale4(net, half), (w, h), interpolation=cv2.INTER_AREA)
