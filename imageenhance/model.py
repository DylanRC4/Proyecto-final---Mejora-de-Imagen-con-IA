"""CNN residual para eliminar ruido, entrenada desde cero por nosotros.

Arquitectura inspirada en DnCNN (Zhang et al., 2017, "Beyond a Gaussian Denoiser"), reducida
para CPU y sin BatchNorm (lotes pequeños). No se usan pesos preentrenados.

Aprendizaje residual: la red no dibuja la imagen limpia; estima el RUIDO y se lo resta:
    limpia ≈ ruidosa − f(ruidosa)
Cada capa Conv2d es la misma operación que nuestra conv2d de NumPy (Sesión 01/04), pero con
kernels que se aprenden por descenso de gradiente en vez de diseñarse a mano, y con muchas
neuronas por capa: cada filtro es un perceptrón (suma ponderada + sesgo, Sesión 11) aplicado en
todas las posiciones de la imagen. Parámetros por capa: (k·k·C_in + 1)·C_out, la misma fórmula
(entradas + 1)·neuronas de la MLP (Sesión 12), pero compartiendo pesos entre posiciones.
Receptivo: depth capas 3x3 -> ventana de (2·depth + 1) píxeles.
"""
import numpy as np
import torch
from torch import nn


class DenoiseCNN(nn.Module):
    def __init__(self, channels: int = 3, features: int = 32, depth: int = 7):
        super().__init__()
        self.config = {"channels": channels, "features": features, "depth": depth}
        conv = lambda cin, cout: nn.Conv2d(cin, cout, 3, padding=1, padding_mode="reflect")
        layers = [conv(channels, features), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [conv(features, features), nn.ReLU(inplace=True)]
        layers.append(conv(features, channels))
        self.noise_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.noise_net(x)


def formula_params(channels: int = 3, features: int = 32, depth: int = 7, k: int = 3) -> int:
    per = lambda cin, cout: (k * k * cin + 1) * cout
    return per(channels, features) + (depth - 2) * per(features, features) + per(features, channels)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC -> 1CHW. PyTorch usa canales primero (Sesión 02: el orden de los ejes importa)."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32))[None]


@torch.no_grad()
def denoise(model: nn.Module, img: np.ndarray, tile: int = 512, pad: int = 16) -> np.ndarray:
    """Aplica la red a una imagen HWC float [0, 1] en el mismo orden de canales del entrenamiento (BGR).

    Las fotos grandes se procesan por bloques con margen `pad` (mayor que el radio receptivo),
    así el resultado es igual al de procesar la imagen completa pero sin agotar la RAM.
    """
    model.eval()
    h, w = img.shape[:2]
    out = np.empty_like(img, dtype=np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0, y1, x1 = max(y - pad, 0), max(x - pad, 0), min(y + tile + pad, h), min(x + tile + pad, w)
            res = model(to_tensor(img[y0:y1, x0:x1]))[0].numpy().transpose(1, 2, 0)
            th, tw = min(tile, h - y), min(tile, w - x)
            out[y:y + th, x:x + tw] = res[y - y0:y - y0 + th, x - x0:x - x0 + tw]
    return np.clip(out, 0.0, 1.0)


def save(model: DenoiseCNN, path, **meta) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), "meta": meta}, path)


def load(path, device: str = "cpu") -> tuple[DenoiseCNN, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)
    model = DenoiseCNN(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck.get("meta", {})
