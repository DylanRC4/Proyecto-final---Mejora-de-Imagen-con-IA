"""Descarga DIV2K (Agustsson y Timofte, 2017) y lo prepara para entrenar la CNN de detalle v2.

DIV2K: 800 fotos oficiales de entrenamiento y 100 de validación, en 2K y alta calidad, de uso
académico. Es el conjunto estándar para restauración y superresolución.

Preparación:
- Cada PNG se lee directamente del .zip (no hace falta descomprimir los 3,9 GB).
- Se reduce ×2 con INTER_AREA: queda de unos 1020 px de lado, la escala a la que trabaja la app
  (máximo 1280 px), y ocupa 4 veces menos memoria al entrenar.
- Split POR FOTOGRAFÍA: 0001-0700 → train, 0701-0800 → val, 0801-0900 (validación oficial) → test.

Uso:  python scripts/prepare_div2k.py
Después se puede borrar data/raw/div2k_zip para liberar espacio.
"""
import hashlib
import json
import sys
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import data  # noqa: E402

BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ZIPS = ["DIV2K_train_HR.zip", "DIV2K_valid_HR.zip"]
ZIP_DIR = data.RAW_ROOT / "div2k_zip"
OUT, MANIFEST = data.DATASETS["div2k"][:2]


def download(name: str) -> Path:
    """Descarga por bloques de 8 MB. Si se corta, al volver a ejecutar continúa donde quedó."""
    dst, tmp = ZIP_DIR / name, ZIP_DIR / (name + ".part")
    head = urllib.request.Request(BASE + name, method="HEAD", headers={"User-Agent": "ImageEnhance-AI"})
    with urllib.request.urlopen(head, timeout=120) as r:
        size = int(r.headers["Content-Length"])
    if dst.exists() and dst.stat().st_size == size:
        print(f"{name}: ya descargado ({size / 1e9:.2f} GB)")
        return dst
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    done = tmp.stat().st_size if tmp.exists() else 0
    if done == size:  # se descargó completo pero no alcanzó a renombrarse
        tmp.replace(dst)
        return dst
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "ImageEnhance-AI", "Range": f"bytes={done}-"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:  # el servidor no aceptó continuar: empieza de cero
            done = 0
        with open(tmp, "ab" if done else "wb") as f:
            while chunk := r.read(8 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r{name}: {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print()
    if done != size:
        raise RuntimeError(f"{name}: descarga incompleta ({done} de {size} bytes). Vuelve a ejecutar para continuar.")
    tmp.replace(dst)
    return dst


def split_of(idx: int) -> str:
    return "train" if idx <= 700 else "val" if idx <= 800 else "test"


def main() -> None:
    manifest = {"dataset": "DIV2K", "source": BASE, "cita": "Agustsson y Timofte, NTIRE 2017",
                "procesado": "reducción ×2 con INTER_AREA, PNG", "splits": {s: [] for s in data.SPLITS}, "sha256": {}}
    for name in ZIPS:
        zpath = download(name)
        with zipfile.ZipFile(zpath) as z:
            entries = sorted(e for e in z.namelist() if e.lower().endswith(".png"))
        local = threading.local()  # un lector del zip por hilo: no se carga el zip entero en RAM

        def process(entry):
            if not hasattr(local, "zip"):
                local.zip = zipfile.ZipFile(zpath)
            stem = Path(entry).stem
            split = split_of(int(stem))
            out = OUT / split / f"{stem}.png"
            if not out.exists():
                img = cv2.imdecode(np.frombuffer(local.zip.read(entry), np.uint8), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_bytes(cv2.imencode(".png", small)[1].tobytes())
                tmp.replace(out)  # si se interrumpe, nunca queda un PNG a medias
            return split, out.name, hashlib.sha256(out.read_bytes()).hexdigest()

        with ThreadPoolExecutor(6) as pool:
            for i, (split, fname, digest) in enumerate(pool.map(process, entries), 1):
                manifest["splits"][split].append(fname)
                manifest["sha256"][f"{split}/{fname}"] = digest
                if i % 100 == 0 or i == len(entries):
                    print(f"{name}: {i}/{len(entries)} fotos procesadas", flush=True)
    manifest["splits"] = {k: sorted(v) for k, v in manifest["splits"].items()}
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), "utf-8")
    print("Listo:", {k: len(v) for k, v in manifest["splits"].items()}, "->", OUT)


if __name__ == "__main__":
    main()
