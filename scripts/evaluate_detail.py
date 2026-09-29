"""Evalúa la CNN de DETALLE en las fotos de PRUEBA contra la foto sin restaurar, la nitidez clásica
y Real-ESRGAN (externo, generativo). Todos los métodos reciben exactamente la misma foto degradada.

Condiciones (todas deterministas):
  fija       desenfoque 1.0 → reducción ×2 → JPEG 35 (la misma de la validación)
  desenfoque desenfoque gaussiano σ = 1.5
  jpeg       compresión JPEG calidad 20
  baja_res   reducción ×2 y ampliación bicúbica
  fuerte     desenfoque 3.0 → reducción ×2 → JPEG 60 (como una foto de celular muy borrosa)
  celular    ruido de cámara (a = 0.003, semilla por foto) → JPEG 85
La intensidad de la nitidez clásica se elige en VALIDACIÓN (30 fotos), como se hizo con los filtros de ruido.

Uso:  python scripts/evaluate_detail.py [--model models/detail_cnn_v2.pt] [--dataset bsds500|div2k] [--limit N]
Salidas: results/detail_eval/<modelo>_<dataset>.md/.json/.csv y results/detail_eval/examples_<modelo>_<dataset>/
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import data, enhance, external, metrics, model as M, noise  # noqa: E402
from imageenhance.io_utils import bgr_to_rgb  # noqa: E402

CONDITIONS = {"fija": lambda x, n: noise.fixed_detail(x), "desenfoque": lambda x, n: noise.add_blur(x, 1.5),
              "jpeg": lambda x, n: noise.add_jpeg(x, 20), "baja_res": lambda x, n: noise.add_downscale(x, 2.0),
              "fuerte": lambda x, n: noise.add_jpeg(noise.add_downscale(noise.add_blur(x, 3.0), 2.0), 60),
              "celular": lambda x, n: noise.add_jpeg(noise.degrade(x, "camara", 0.003, data.noise_seed(n, "camara", 0.003)), 85)}
AMOUNTS = (0.3, 0.6, 1.0, 1.5)
EXAMPLES = 2


def best_amount(degrade, imgs):
    scores = {a: np.mean([metrics.psnr(im, enhance.unsharp(degrade(im, n), a)) for n, im in imgs]) for a in AMOUNTS}
    return max(scores, key=scores.get)


def save_example(path, clean, outs, scores, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(outs) + 1, figsize=(4 * (len(outs) + 1), 3.6))
    for ax, (label, im) in zip(axes, [("original limpia", clean)] + list(outs.items())):
        ax.imshow(np.clip(bgr_to_rgb(im), 0, 1))
        extra = "" if label.startswith("original") else f"\n{scores[label]['psnr']:.2f} dB | SSIM {scores[label]['ssim']:.3f}"
        ax.set_title(label + extra, fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 88})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detail_cnn.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val", type=int, default=30, help="fotos de validación para elegir la nitidez clásica")
    ap.add_argument("--dataset", default="bsds500", choices=["bsds500", "div2k"])
    args = ap.parse_args()
    root = data.ROOT
    net, meta = M.load(root / args.model)
    esrgan = external.load_esrgan() if external.ESRGAN_PATH.exists() else None
    val = data.load_split("val", args.val, dataset=args.dataset)
    test = data.load_split("test", args.limit, dataset=args.dataset)
    tag = f"{Path(args.model).stem}_{args.dataset}"
    out = root / "results" / "detail_eval"
    ex_dir = out / f"examples_{tag}"
    ex_dir.mkdir(parents=True, exist_ok=True)
    rows, summary, amounts, t0 = [], {}, {}, time.time()
    for cond, degrade in CONDITIONS.items():
        amounts[cond] = best_amount(degrade, val)
        for idx, (name, clean) in enumerate(test):
            deg = degrade(clean, name).astype(np.float32)
            outs = {"sin_restaurar": deg, "nitidez_clasica": enhance.unsharp(deg, amounts[cond]), "cnn_detalle": M.denoise(net, deg)}
            if esrgan is not None:
                outs["real_esrgan"] = external.restore(esrgan, deg)
            scores = {m: metrics.evaluate(clean, o) for m, o in outs.items()}
            rows += [{"condicion": cond, "imagen": name, "metodo": m, "psnr": round(s["psnr"], 4), "ssim": round(s["ssim"], 4)}
                     for m, s in scores.items()]
            if idx < EXAMPLES:
                save_example(ex_dir / f"{cond}_{Path(name).stem}.jpg", clean, outs, scores, f"{name} — {cond}")
        c = [r for r in rows if r["condicion"] == cond]
        methods = list(dict.fromkeys(r["metodo"] for r in c))
        res = {m: {k: float(np.mean([r[k] for r in c if r["metodo"] == m])) for k in ("psnr", "ssim")} for m in methods}
        cnn = {r["imagen"]: r["psnr"] for r in c if r["metodo"] == "cnn_detalle"}
        base = {r["imagen"]: r["psnr"] for r in c if r["metodo"] == "sin_restaurar"}
        res["cnn_mejora_la_foto_en"] = f"{sum(cnn[i] > base[i] for i in cnn)}/{len(cnn)}"
        summary[cond] = res
        print(f"{cond:10s}" + "".join(f" | {m} {res[m]['psnr']:.2f}/{res[m]['ssim']:.3f}" for m in methods)
              + f" | CNN mejora {res['cnn_mejora_la_foto_en']}", flush=True)
    with open(out / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    info = {"model": args.model, "dataset": args.dataset, "model_epoch": meta.get("epoch"), "test_images": len(test),
            "nitidez_clasica_elegida_en_validacion": amounts, "seconds": round(time.time() - t0, 1), "results": summary}
    (out / f"{tag}.json").write_text(json.dumps(info, indent=1), "utf-8")
    methods = list(next(iter(summary.values())).keys())[:-1]
    lines = ["| Condición | " + " | ".join(methods) + " | CNN mejora la foto |", "|" + "---|" * (len(methods) + 2)]
    for cond, res in summary.items():
        lines.append(f"| {cond} | " + " | ".join(f"{res[m]['psnr']:.2f} dB / {res[m]['ssim']:.3f}" for m in methods)
                     + f" | {res['cnn_mejora_la_foto_en']} |")
    (out / f"{tag}.md").write_text(
        f"{args.model} en {len(test)} fotos de prueba ({args.dataset}). PSNR medio / SSIM medio. Real-ESRGAN es externo y generativo.\n\n"
        + "\n".join(lines) + "\n", "utf-8")
    print(f"{len(test)} fotos de prueba en {time.time() - t0:.0f} s -> results/detail_eval/{tag}.md")


if __name__ == "__main__":
    main()
