"""Línea base clásica: ajusta mediana y gaussiano en VALIDACIÓN y guarda los mejores parámetros.

Para que la comparación con la CNN sea justa, los filtros clásicos también eligen sus
hiperparámetros (tamaño de kernel, sigma) con datos de validación, nunca con los de prueba.
Criterio de selección: PSNR medio. Resultado: results/classic_params.json y classic_val.csv.

Uso:  python scripts/compare_classic.py [--limit N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import classic, data, metrics, noise  # noqa: E402

CONDITIONS = [("gaussian", 15), ("gaussian", 25), ("gaussian", 50), ("salt_pepper", 0.05)]
GRID = [("median", {"ksize": k}) for k in (3, 5, 7)] + \
       [("gaussian", {"ksize": 2 * int(np.ceil(3 * s)) + 1, "sigma": s}) for s in (0.5, 0.8, 1.1, 1.5, 2.0, 2.5)]
OUT = data.ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    imgs = data.load_split("val", args.limit)
    rows, best, t0 = [], {}, time.time()
    for kind, level in CONDITIONS:
        noisy = [noise.degrade(im, kind, level, data.noise_seed(n, kind, level)) for n, im in imgs]
        base = [metrics.evaluate(im, nz) for (_, im), nz in zip(imgs, noisy)]
        rows.append({"noise": kind, "level": level, "method": "sin_filtro", "params": "{}",
                     "psnr": np.mean([m["psnr"] for m in base]), "ssim": np.mean([m["ssim"] for m in base])})
        for method, params in GRID:
            ms = [metrics.evaluate(im, classic.apply(nz, method, **params)) for (_, im), nz in zip(imgs, noisy)]
            rows.append({"noise": kind, "level": level, "method": method, "params": json.dumps(params),
                         "psnr": np.mean([m["psnr"] for m in ms]), "ssim": np.mean([m["ssim"] for m in ms])})
        cond = [r for r in rows if r["noise"] == kind and r["level"] == level]
        for method in ("median", "gaussian"):
            r = max((r for r in cond if r["method"] == method), key=lambda r: r["psnr"])
            best[f"{kind}_{level}_{method}"] = json.loads(r["params"])
            print(f"{kind:11s} {level:<5} {method:8s} {r['params']:28s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.4f}"
                  f"   (sin filtro {cond[0]['psnr']:.2f} / {cond[0]['ssim']:.4f})", flush=True)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "classic_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows({**r, "psnr": f"{r['psnr']:.4f}", "ssim": f"{r['ssim']:.4f}"} for r in rows)
    meta = {"split": "val", "images": len(imgs), "criterion": "psnr", "best": best}
    (OUT / "classic_params.json").write_text(json.dumps(meta, indent=1), "utf-8")
    print(f"{len(imgs)} imágenes de validación, {time.time() - t0:.0f} s -> results/classic_params.json")


if __name__ == "__main__":
    main()
