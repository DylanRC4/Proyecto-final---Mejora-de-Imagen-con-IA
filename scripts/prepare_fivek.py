"""Prepara MIT-Adobe FiveK (versión 480p del paper 3D-LUT) para aprender el retoque del EXPERTO C.

Fuente: Bychkovsky et al., 2011 (MIT y Adobe), solo para investigación. Se descarga a mano en
data/raw/fivek/ (ignorado por Git). Se usan las entradas sRGB de 8 bits y los retoques del experto C.
Split por FOTOGRAFÍA: si vienen las listas del paper 3D-LUT (test.txt, train_input.txt, train_label.txt) se
usa su prueba oficial; si no, las últimas 500 por nombre. Del resto se separan 500 al azar (semilla fija)
para validación.

Uso:  python scripts/prepare_fivek.py
Crea data/fivek_splits.json (versionado) y data/processed/fivek_{train,val}.npz (parches 192×192, uint8).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fivek"
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SIZE = 192


def find_dirs():
    """Busca la carpeta de entradas sRGB y la del experto C, prefiriendo las de 480p."""
    dirs = {p.parent for p in RAW.rglob("*") if p.suffix.lower() in EXT}
    score = lambda d: ("480" in str(d)) - ("xyz" in str(d).lower()) - ("16" in d.name)
    expert = [d for d in dirs if "expertc" in str(d).lower().replace("_", "")]
    inputs = [d for d in dirs if "input" in str(d).lower() and "xyz" not in str(d).lower() and d not in expert]
    if not expert or not inputs:
        sys.exit(f"No encontré entradas y experto C dentro de {RAW}. Carpetas con imágenes: {sorted(map(str, dirs))[:10]}")
    return max(inputs, key=score), max(expert, key=score)


def load(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)  # 8 bits BGR (como el resto del proyecto)


def crop(img: np.ndarray) -> np.ndarray:
    """Lado corto a 192 px y recorte central: para aprender ajustes GLOBALES basta la foto reducida."""
    h, w = img.shape[:2]
    s = SIZE / min(h, w)
    img = cv2.resize(img, (max(SIZE, round(w * s)), max(SIZE, round(h * s))), interpolation=cv2.INTER_AREA)
    y, x = (img.shape[0] - SIZE) // 2, (img.shape[1] - SIZE) // 2
    return img[y:y + SIZE, x:x + SIZE]


def main() -> None:
    din, dexp = find_dirs()
    stem = lambda d: {p.stem: p for p in d.iterdir() if p.suffix.lower() in EXT}
    a, b = stem(din), stem(dexp)
    names = sorted(set(a) & set(b))
    print(f"Entradas: {din}\nExperto C: {dexp}\nPares encontrados: {len(names)}")
    rng = np.random.default_rng(2026)
    listed = RAW / "test.txt"
    k = min(500, len(names) // 10)
    test = sorted(set(listed.read_text().split()) & set(names)) if listed.exists() else names[-k:]
    rest = sorted(set(names) - set(test))
    val = sorted(rng.choice(rest, size=min(500, len(rest) // 9), replace=False).tolist())
    train = sorted(set(rest) - set(val))
    manifest = {"dataset": "MIT-Adobe FiveK 480p (entradas sRGB, experto C)", "cita": "Bychkovsky et al., CVPR 2011",
                "licencia": "Solo investigación (Adobe / MIT)", "input_dir": str(din.relative_to(ROOT)),
                "expert_dir": str(dexp.relative_to(ROOT)), "splits": {"train": train, "val": val, "test": test}}
    (ROOT / "data" / "fivek_splits.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), "utf-8")
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for split, ns in (("train", train), ("val", val)):
        X = np.stack([crop(load(a[n])) for n in ns])
        Y = np.stack([crop(load(b[n])) for n in ns])
        np.savez(out / f"fivek_{split}.npz", X=X, Y=Y)
        print(f"{split}: {len(ns)} pares -> data/processed/fivek_{split}.npz ({X.nbytes * 2 / 1e6:.0f} MB)")
    print(f"test: {len(test)} fotos (se evalúan completas, a 480p)")


if __name__ == "__main__":
    main()
