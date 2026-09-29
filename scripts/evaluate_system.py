"""Evalúa el SISTEMA COMPLETO en modo automático (diagnóstico MLP → plan → pasos) en fotos de PRUEBA
con degradaciones combinadas, como llegan las fotos reales. Nadie elige pasos ni parámetros.

Condiciones (deterministas; orden de una cámara real: lente → exposición → sensor → JPEG):
  limpia           sin degradar: el sistema NO debería estropearla (mide falsos positivos)
  ruido            ruido gaussiano σ 15
  oscura_ruidosa   −2 EV y ruido σ 10 (las fotos oscuras suelen tener ruido)
  desenfoque_jpeg  desenfoque σ 1.2 y JPEG calidad 40
  oscura_velo      −2 EV y velo (negros levantados, como un lente sucio)
  dominante        luz cálida: ganancias B, G, R = 0.8, 1.0, 1.15
  todo             desenfoque 1.0 → −1.5 EV → ruido σ 10 → JPEG 50
  celular          ruido de cámara (a = 0.003) → JPEG 85, como una foto de celular con poca luz
  fuerte           desenfoque 3.0 → reducción ×2 → JPEG 60, una foto de celular muy borrosa
Métodos: sin_procesar | sistema (nuestras CNN) | sistema_clasico (mismo plan, pero ruido con filtro
gaussiano elegido en validación y detalle con máscara de desenfoque: muestra cuánto aportan las CNN).
El PSNR de una foto idéntica a la limpia es infinito: se limita a 60 dB para poder promediar.

Uso:  python scripts/evaluate_system.py [--split val] [--noise models/denoise_cnn_v2.pt] [--detail models/detail_cnn_v3.pt] [--limit N]
Con --split val se ELIGEN modelos y reglas (100 fotos de validación); con prueba (por defecto) solo se confirman.
Salidas: results/system_eval/<dataset>[_val]_<ruido>_<detalle>.md/.json/.csv y examples_<...>/
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import classic, data, diagnosis, metrics, model as M, noise, pipeline  # noqa: E402
from scripts.evaluate_detail import save_example  # noqa: E402

G = lambda x, s, name, c: noise.add_gaussian(x, s, seed=data.noise_seed(name, c, s))
CONDITIONS = {
    "limpia": lambda x, n: x,
    "ruido": lambda x, n: G(x, 15, n, "ruido"),
    "oscura_ruidosa": lambda x, n: G(noise.add_exposure(x, -2), 10, n, "oscura_ruidosa"),
    "desenfoque_jpeg": lambda x, n: noise.add_jpeg(noise.add_blur(x, 1.2), 40),
    "oscura_velo": lambda x, n: 0.9 * noise.add_exposure(x, -2) + 0.06,
    "dominante": lambda x, n: noise.add_cast(x),
    "todo": lambda x, n: noise.add_jpeg(G(noise.add_exposure(noise.add_blur(x, 1.0), -1.5), 10, n, "todo"), 50),
    "celular": lambda x, n: noise.add_jpeg(noise.degrade(x, "camara", 0.003, data.noise_seed(n, "camara", 0.003)), 85),
    "fuerte": lambda x, n: noise.add_jpeg(noise.add_downscale(noise.add_blur(x, 3.0), 2.0), 60),
}
EXAMPLES = 2


def score(clean, img):
    return min(metrics.psnr(clean, img), 60.0), metrics.ssim(clean, img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bsds500", choices=["bsds500", "div2k"])
    ap.add_argument("--detail", default=f"models/{pipeline.DETAIL_MODEL}", help="CNN de detalle que usa la app")
    ap.add_argument("--noise", default=f"models/{pipeline.NOISE_MODEL}", help="CNN de ruido que usa la app")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = data.ROOT
    models = {"denoiser": M.load(root / args.noise)[0], "detail": M.load(root / args.detail)[0]}
    tag = f"{args.dataset}{'_val' if args.split == 'val' else ''}_{Path(args.noise).stem}_{Path(args.detail).stem}"
    diag = diagnosis.Diagnoser(root / "models" / "diagnosis_mlp.pt")
    p = root / "results" / "classic_params.json"
    gauss = json.loads(p.read_text("utf-8"))["best"].get("gaussian_15_gaussian", {}) if p.exists() else {}
    test = data.load_split(args.split, args.limit, dataset=args.dataset)
    out = root / "results" / "system_eval"
    ex_dir = out / f"examples_{tag}"
    ex_dir.mkdir(parents=True, exist_ok=True)
    rows, summary, t0 = [], {}, time.time()
    for cond, degrade in CONDITIONS.items():
        plans, secs = Counter(), []
        for idx, (name, clean) in enumerate(test):
            deg = degrade(clean, name).astype(np.float32)
            t = time.perf_counter()
            steps = pipeline.plan(diag(deg)["problemas"])
            ours = pipeline.run(deg, steps, models)[0]
            secs.append(time.perf_counter() - t)
            clas = classic.gaussian_filter(deg, **gauss) if "ruido" in steps else deg
            clas = pipeline.run(clas, [s for s in steps if s != "ruido"], {"detail": None})[0]  # detalle -> máscara de desenfoque
            plans[" → ".join(steps) or "(nada)"] += 1
            outs = {"sin_procesar": deg, "sistema": ours, "sistema_clasico": clas}
            for m, o in outs.items():
                ps, ss = score(clean, o)
                rows.append({"condicion": cond, "imagen": name, "metodo": m, "pasos": " → ".join(steps), "psnr": round(ps, 4), "ssim": round(ss, 4)})
            if idx < EXAMPLES:
                save_example(ex_dir / f"{cond}_{Path(name).stem}.jpg", clean, outs,
                             {m: dict(zip(("psnr", "ssim"), score(clean, o))) for m, o in outs.items()}, f"{name} — {cond}: {' → '.join(steps) or 'sin pasos'}")
        c = [r for r in rows if r["condicion"] == cond]
        res = {m: {k: float(np.mean([r[k] for r in c if r["metodo"] == m])) for k in ("psnr", "ssim")} for m in outs}
        base = {r["imagen"]: r["psnr"] for r in c if r["metodo"] == "sin_procesar"}
        sis = {r["imagen"]: r["psnr"] for r in c if r["metodo"] == "sistema"}
        res["mejora"] = f"{sum(sis[i] > base[i] + 0.05 for i in sis)}/{len(sis)}"
        res["empeora"] = f"{sum(sis[i] < base[i] - 0.05 for i in sis)}/{len(sis)}"
        res["planes_mas_comunes"] = dict(plans.most_common(3))
        res["segundos_por_foto"] = round(float(np.mean(secs)), 3)
        summary[cond] = res
        print(f"{cond:15s} | sin procesar {res['sin_procesar']['psnr']:.2f}/{res['sin_procesar']['ssim']:.3f}"
              f" | sistema {res['sistema']['psnr']:.2f}/{res['sistema']['ssim']:.3f}"
              f" | clásico {res['sistema_clasico']['psnr']:.2f}/{res['sistema_clasico']['ssim']:.3f}"
              f" | mejora {res['mejora']} empeora {res['empeora']} | {res['segundos_por_foto']} s/foto | {dict(plans.most_common(2))}", flush=True)
    with open(out / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    info = {"dataset": args.dataset, "split": args.split, "noise_model": args.noise, "detail_model": args.detail, "test_images": len(test), "clasico_ruido": gauss,
            "seconds": round(time.time() - t0, 1), "results": summary}
    (out / f"{tag}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), "utf-8")
    lines = ["| Condición | Sin procesar | Sistema (nuestras CNN) | Sistema clásico | Mejora / empeora | Plan más común |", "|---|---|---|---|---|---|"]
    for cond, r in summary.items():
        lines.append(f"| {cond} | " + " | ".join(f"{r[m]['psnr']:.2f} dB / {r[m]['ssim']:.3f}" for m in ("sin_procesar", "sistema", "sistema_clasico"))
                     + f" | {r['mejora']} / {r['empeora']} | {next(iter(r['planes_mas_comunes']))} |")
    (out / f"{tag}.md").write_text(
        f"Sistema automático en {len(test)} fotos de {'validación' if args.split == 'val' else 'prueba'} ({args.dataset}), CNN de ruido {args.noise} y de detalle {args.detail}. "
        "PSNR medio (limitado a 60 dB) / SSIM medio. Mejora o empeora: cambio de PSNR mayor de 0,05 dB.\n\n" + "\n".join(lines) + "\n", "utf-8")
    print(f"{len(test)} fotos × {len(CONDITIONS)} condiciones en {time.time() - t0:.0f} s -> results/system_eval/{tag}.md")


if __name__ == "__main__":
    main()
