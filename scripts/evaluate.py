"""Evaluación final en el conjunto de PRUEBA (fotos que ni la CNN ni el ajuste de filtros vieron).

Compara, con el mismo ruido (semilla por imagen y condición):
  sin filtro | mediana (mejor en validación) | gaussiano (mejor en validación) | CNN
Condiciones: gaussiano sigma 15, 25, 50, sal y pimienta 5 % (ninguna CNN se entrenó con este ruido: muestra
honestamente sus límites) y ruido de CÁMARA (noise.add_camera_noise, a = 0.003), el que se parece al del celular.
Para el ruido de cámara los filtros clásicos usan los parámetros elegidos en validación para gaussiano sigma 15.

Uso:  python scripts/evaluate.py [--model models/denoise_cnn_v2.pt] [--limit N]
Salidas: results/noise_eval/<modelo>.md/.json/.csv y results/noise_eval/examples_<modelo>/*.png
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import classic, data, metrics, model as M, noise  # noqa: E402
from imageenhance.io_utils import bgr_to_rgb  # noqa: E402

CONDITIONS = [("gaussian", 15), ("gaussian", 25), ("gaussian", 50), ("salt_pepper", 0.05), ("camara", 0.003)]
METHODS = ["sin_filtro", "mediana", "gaussiano", "cnn"]
EXAMPLES = 3


def save_example(path, clean, outs, scores, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, (label, im) in zip(axes, [("limpia (referencia)", clean)] + list(outs.items())):
        ax.imshow(bgr_to_rgb(im))
        extra = "" if label.startswith("limpia") else f"\nPSNR {scores[label]['psnr']:.2f} dB | SSIM {scores[label]['ssim']:.3f}"
        ax.set_title(label + extra, fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/denoise_cnn.pt")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = data.ROOT
    net, meta = M.load(root / args.model)
    best = json.loads((root / "results" / "classic_params.json").read_text("utf-8"))["best"]
    imgs = data.load_split("test", args.limit)
    tag = Path(args.model).stem
    out = root / "results" / "noise_eval"
    ex_dir = out / f"examples_{tag}"
    ex_dir.mkdir(parents=True, exist_ok=True)
    rows, summary, t0 = [], {}, time.time()
    for kind, level in CONDITIONS:
        med_p = best.get(f"{kind}_{level}_median", best["gaussian_15_median"])
        gau_p = best.get(f"{kind}_{level}_gaussian", best["gaussian_15_gaussian"])
        for idx, (name, clean) in enumerate(imgs):
            noisy = noise.degrade(clean, kind, level, data.noise_seed(name, kind, level))
            outs = {"sin_filtro": noisy, "mediana": classic.median_filter(noisy, **med_p),
                    "gaussiano": classic.gaussian_filter(noisy, **gau_p), "cnn": M.denoise(net, noisy)}
            scores = {m: metrics.evaluate(clean, o) for m, o in outs.items()}
            rows += [{"noise": kind, "level": level, "image": name, "method": m,
                      "psnr": round(s["psnr"], 4), "ssim": round(s["ssim"], 4)} for m, s in scores.items()]
            if idx < EXAMPLES:
                save_example(ex_dir / f"{kind}_{level}_{Path(name).stem}.png", clean, outs, scores,
                             f"{name} — ruido {kind} {level}")
        cond = [r for r in rows if r["noise"] == kind and r["level"] == level]
        res = {m: {k: float(np.mean([r[k] for r in cond if r["method"] == m])) for k in ("psnr", "ssim")} for m in METHODS}
        cnn = {r["image"]: r["psnr"] for r in cond if r["method"] == "cnn"}
        gau = {r["image"]: r["psnr"] for r in cond if r["method"] == "gaussiano"}
        res["cnn_gana_a_gaussiano_en"] = f"{sum(cnn[i] > gau[i] for i in cnn)}/{len(cnn)}"
        summary[f"{kind}_{level}"] = res
        print(f"{kind:11s} {level:<5}" + "".join(f" | {m} {res[m]['psnr']:.2f}/{res[m]['ssim']:.3f}" for m in METHODS)
              + f" | CNN>gauss en {res['cnn_gana_a_gaussiano_en']}", flush=True)

    with open(out / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    info = {"model": args.model, "model_epoch": meta.get("epoch"), "test_images": len(imgs),
            "classic_params": best, "seconds": round(time.time() - t0, 1), "results": summary}
    (out / f"{tag}.json").write_text(json.dumps(info, indent=1), "utf-8")
    lines = ["| Ruido | Sin filtro | Mediana | Gaussiano | CNN (nuestra) | CNN > gaussiano |", "|---|---|---|---|---|---|"]
    for cond, res in summary.items():
        lines.append(f"| {cond} | " + " | ".join(f"{res[m]['psnr']:.2f} dB / {res[m]['ssim']:.3f}" for m in METHODS)
                     + f" | {res['cnn_gana_a_gaussiano_en']} |")
    (out / f"{tag}.md").write_text(
        f"{args.model} en el conjunto de prueba BSDS500: {len(imgs)} fotos. Valores: PSNR medio / SSIM medio.\n\n" + "\n".join(lines) + "\n", "utf-8")
    print(f"{len(imgs)} fotos de prueba en {time.time() - t0:.0f} s -> results/noise_eval/{tag}.md")


if __name__ == "__main__":
    main()
