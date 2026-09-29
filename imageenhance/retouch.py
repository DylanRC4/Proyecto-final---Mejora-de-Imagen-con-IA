"""Retoque APRENDIDO (tono y color) con MIT-Adobe FiveK: aprende cómo retoca un fotógrafo (experto C).

Una CNN pequeña mira la foto reducida a 128×128 y predice solo 13 números: ganancias de color (3),
exposición (1), una curva de tono de 8 tramos (8) y saturación (1). Esos ajustes GLOBALES se aplican
con fórmulas fijas a la foto completa: cada píxel pasa por la misma curva, así que es imposible que
invente detalle (no es generativo) y el resultado es explicable ("subió 0,4 EV y enfrió el color").
La última capa arranca en cero: la red nueva devuelve la foto sin cambios (como las CNN de restauración).
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KNOTS = 8
N_PARAMS = 3 + 1 + KNOTS + 1


class RetouchNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*c(3, width // 2), *c(width // 2, width), *c(width, 2 * width), *c(2 * width, 2 * width))
        self.head = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.ReLU(inplace=True), nn.Linear(2 * width, N_PARAMS))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: lote N×3×H×W en [0, 1] (BGR). Devuelve la foto retocada del mismo tamaño."""
        return adjust(x, self.predict(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        small = F.interpolate(x, size=(128, 128), mode="bilinear", antialias=True, align_corners=False)
        return self.head(self.features(small).mean(dim=(2, 3)))


def adjust(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Aplica los 13 parámetros. Con p = 0 la foto sale idéntica."""
    gains = torch.exp(0.5 * torch.tanh(p[:, 0:3]))[:, :, None, None]            # balance de blancos: ×0,61 a ×1,65
    ev = 2.0 * torch.tanh(p[:, 3])[:, None, None, None]                          # exposición: −2 a +2 EV
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)    # sRGB -> luz lineal
    lin = (lin * gains * 2.0 ** ev).clamp(0, 1)
    y = torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
    steps = torch.softmax(p[:, 4:4 + KNOTS], dim=1)[:, None, :, None, None]     # curva de tono: 8 tramos que suman 1
    seg = torch.arange(KNOTS, device=x.device, dtype=x.dtype)[None, None, :, None, None]
    y = (steps * (KNOTS * y.clamp(0, 1)[:, :, None] - seg).clamp(0, 1)).sum(2)  # monótona; tramos iguales = identidad
    lum = (y * torch.tensor([0.114, 0.587, 0.299], device=y.device)[None, :, None, None]).sum(1, keepdim=True)
    sat = torch.exp(0.5 * torch.tanh(p[:, -1]))[:, None, None, None]            # saturación: ×0,61 a ×1,65
    return (lum + sat * (y - lum)).clamp(0, 1)


def describe(p: np.ndarray) -> str:
    g = np.exp(0.5 * np.tanh(p[0:3]))
    return (f"ganancias B, G, R = {g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}; exposición {2 * np.tanh(p[3]):+.2f} EV; "
            f"saturación ×{np.exp(0.5 * np.tanh(p[-1])):.2f}")


@torch.no_grad()
def retouch(model: RetouchNet, img: np.ndarray) -> tuple[np.ndarray, str]:
    """Foto BGR float [0, 1] de cualquier tamaño: la red mira una versión de 128×128 y los ajustes
    se aplican a la foto completa."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32))[None]
    small = torch.from_numpy(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).transpose(2, 0, 1).copy())[None]
    p = model.head(model.features(small).mean(dim=(2, 3)))
    return adjust(x, p)[0].numpy().transpose(1, 2, 0), describe(p[0].numpy())


def load(path) -> RetouchNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = RetouchNet(ck.get("width", 32))
    net.load_state_dict(ck["state_dict"])
    return net.eval()
