"""Acceso al dataset (split por imagen) y semillas reproducibles para el ruido.

- Dos datasets, ambos con split POR FOTOGRAFÍA:
  "bsds500" (data/splits.json, scripts/prepare_data.py) y "div2k" (data/div2k_splits.json, scripts/prepare_div2k.py).
- La semilla del ruido de cada imagen de validación/prueba se deriva de su nombre, del tipo de
  ruido y del nivel con CRC32 (estable entre ejecuciones y sistemas; hash() de Python no lo es).
  Así, "12003.jpg con sigma=25" tiene siempre exactamente el mismo ruido.
"""
import json
import zlib
from pathlib import Path

import numpy as np

from .io_utils import load_bgr, to_float

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
DATASETS = {"bsds500": (RAW_ROOT / "bsds500", ROOT / "data" / "splits.json", "scripts/prepare_data.py"),
            "div2k": (RAW_ROOT / "div2k", ROOT / "data" / "div2k_splits.json", "scripts/prepare_div2k.py")}
RAW, MANIFEST = DATASETS["bsds500"][:2]
BASE_SEED = 2026
SPLITS = ("train", "val", "test")


def split_names(split: str, dataset: str = "bsds500") -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"Split desconocido: {split}")
    _, manifest, script = DATASETS[dataset]
    if not manifest.exists():
        raise FileNotFoundError(f"Falta {manifest.name}. Ejecuta: python {script}")
    return json.loads(manifest.read_text("utf-8"))["splits"][split]


def split_paths(split: str, limit: int | None = None, dataset: str = "bsds500") -> list[Path]:
    raw, _, script = DATASETS[dataset]
    paths = [raw / split / n for n in split_names(split, dataset)][:limit]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} imágenes. Ejecuta: python {script}")
    return paths


def load_split(split: str, limit: int | None = None, as_float: bool = True, dataset: str = "bsds500") -> list[tuple[str, np.ndarray]]:
    """Lista de (nombre, imagen BGR). as_float=False conserva uint8 (4 veces menos memoria)."""
    return [(p.name, to_float(img) if as_float else img)
            for p in split_paths(split, limit, dataset) for img in [load_bgr(p)]]


def noise_seed(name: str, kind: str, level: float, base: int = BASE_SEED) -> int:
    return zlib.crc32(f"{name}|{kind}|{level}".encode()) ^ base


def random_patches(img: np.ndarray, n: int, size: int, rng: np.random.Generator, augment: bool = True) -> np.ndarray:
    """Recorta n parches aleatorios (n, size, size, C) de UNA imagen, con giros y espejos opcionales."""
    h, w = img.shape[:2]
    ys, xs = rng.integers(0, h - size + 1, n), rng.integers(0, w - size + 1, n)
    out = np.stack([img[y:y + size, x:x + size] for y, x in zip(ys, xs)])
    if augment:
        for i in range(n):
            out[i] = np.rot90(out[i], rng.integers(4))
            if rng.random() < 0.5:
                out[i] = out[i][:, ::-1]
    return out
