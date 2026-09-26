"""Descarga BSDS500 (500 fotos naturales a color) y registra el split oficial por imagen.

Fuente: Berkeley Segmentation Dataset 500 (Arbeláez et al., 2011), uso académico/no comercial.
Se descarga desde el espejo público github.com/BIDS/BSDS500, fijado a un commit concreto.
El split oficial ya separa por FOTOGRAFÍA: train 200, val 100, test 200. Ninguna foto aparece en
dos subconjuntos, así que ningún parche de una imagen de prueba se usa para entrenar.

Uso:  python scripts/prepare_data.py
Crea data/raw/bsds500/{train,val,test}/*.jpg (ignorado por Git) y data/splits.json (versionado).
"""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "BIDS/BSDS500"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bsds500"
MANIFEST = ROOT / "data" / "splits.json"
PREFIX = "BSDS500/data/images/"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ImageEnhance-AI"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def build_manifest() -> dict:
    sha = json.loads(get(f"https://api.github.com/repos/{REPO}/commits/master"))["sha"]
    tree = json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}?recursive=1"))["tree"]
    splits = {"train": [], "val": [], "test": []}
    for item in tree:
        p = item["path"]
        if p.startswith(PREFIX) and p.endswith(".jpg"):
            split, name = p[len(PREFIX):].split("/")
            splits[split].append(name)
    return {"dataset": "BSDS500", "source": f"https://github.com/{REPO}", "commit": sha,
            "license": "Uso académico / no comercial (Berkeley Segmentation Dataset)",
            "splits": {k: sorted(v) for k, v in splits.items()}, "sha256": {}}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else build_manifest()
    base = f"https://raw.githubusercontent.com/{REPO}/{manifest['commit']}/{PREFIX}"
    jobs = [(s, n) for s, names in manifest["splits"].items() for n in names]

    def fetch(job):
        split, name = job
        dst = RAW / split / name
        expected = manifest["sha256"].get(f"{split}/{name}")
        if dst.exists() and (expected is None or hashlib.sha256(dst.read_bytes()).hexdigest() == expected):
            return job, hashlib.sha256(dst.read_bytes()).hexdigest()
        data = get(base + f"{split}/{name}")
        digest = hashlib.sha256(data).hexdigest()
        if expected and digest != expected:
            raise RuntimeError(f"Hash distinto en {split}/{name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return job, digest

    with ThreadPoolExecutor(8) as pool:
        for i, ((split, name), digest) in enumerate(pool.map(fetch, jobs), 1):
            manifest["sha256"][f"{split}/{name}"] = digest
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)

    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), "utf-8")
    sizes = {k: len(v) for k, v in manifest["splits"].items()}
    print("Listo:", sizes, "->", RAW)


if __name__ == "__main__":
    sys.exit(main())
