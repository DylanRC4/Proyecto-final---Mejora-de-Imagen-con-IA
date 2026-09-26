"""Descarga los modelos EXTERNOS preentrenados a models/external/ y verifica su SHA-256.

No se suben a GitHub: no son nuestros y así queda claro qué entrenamos nosotros (models/*.pt)
y qué es externo. Uso:  python scripts/download_models.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance.external import EXTERNAL  # noqa: E402

MODELS = {
    "realesr-general-x4v3.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                                 "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
}


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in MODELS.items():
        dst = EXTERNAL / name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
            print(f"{name}: ya está y el hash coincide")
            continue
        print(f"Descargando {name} ...", flush=True)
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ImageEnhance-AI"}), timeout=300).read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"{name}: el hash no coincide, archivo corrupto o distinto")
        dst.write_bytes(data)
        print(f"{name}: {len(data) / 1e6:.1f} MB, hash verificado")


if __name__ == "__main__":
    main()
