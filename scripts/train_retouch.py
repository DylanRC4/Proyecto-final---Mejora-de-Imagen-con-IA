"""Entrena y evalúa el RETOQUE APRENDIDO (imageenhance/retouch.py) con MIT-Adobe FiveK, experto C.

Pérdida MSE entre la foto retocada por la red y la del experto (así se optimiza lo mismo que mide el
PSNR). Adam con coseno, mejor época por PSNR de VALIDACIÓN. Al final se mide en las 500 fotos de
PRUEBA completas (480p) contra: la entrada sin tocar, nuestras reglas actuales de tono (niveles y gamma
solo si el diagnóstico lo pide) y esas reglas + balance de blancos gray-edge.

Uso:  python scripts/train_retouch.py --bench 20     # mide velocidad, no guarda nada
      python scripts/train_retouch.py                # entrena y evalúa
      python scripts/train_retouch.py --eval-only    # solo la evaluación en prueba
Salidas: models/retouch_fivek.pt, results/retouch_fivek/{train_log.csv, test.json, test.md}
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import diagnosis, enhance, metrics, pipeline, retouch as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL, OUT = ROOT / "models" / "retouch_fivek.pt", ROOT / "results" / "retouch_fivek"


def batches(X, Y, bs, rng=None):
    idx = rng.permutation(len(X)) if rng is not None else np.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        x = torch.from_numpy(X[j]).permute(0, 3, 1, 2).float() / 255
        y = torch.from_numpy(Y[j]).permute(0, 3, 1, 2).float() / 255
        if rng is not None and rng.random() < 0.5:
            x, y = x.flip(3), y.flip(3)
        yield x, y


def val_psnr(net, X, Y):
    net.eval()
    with torch.no_grad():
        mse = torch.cat([((net(x) - y) ** 2).mean(dim=(1, 2, 3)) for x, y in batches(X, Y, 64)])
    return float((10 * torch.log10(1 / mse.clamp_min(1e-10))).mean())


def train(args):
    d = {s: np.load(ROOT / "data" / "processed" / f"fivek_{s}.npz") for s in ("train", "val")}
    (Xtr, Ytr), (Xva, Yva) = [(d[s]["X"][:args.limit], d[s]["Y"][:args.limit]) for s in ("train", "val")]
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    net = R.RetouchNet(args.width)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 100)
    print(f"{len(Xtr)} pares de entrenamiento, {len(Xva)} de validación, {sum(p.numel() for p in net.parameters())} parámetros")
    base = float(np.mean([min(metrics.psnr(y / 255, x / 255), 60.0) for x, y in zip(Xva.astype(np.float32), Yva.astype(np.float32))]))
    print(f"Validación sin retocar: {base:.2f} dB")
    OUT.mkdir(parents=True, exist_ok=True)
    best, log = -math.inf, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total, n = time.time(), 0.0, 0
        for k, (x, y) in enumerate(batches(Xtr, Ytr, args.batch, rng)):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), y)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(x), n + len(x)
            if args.bench and k + 1 == args.bench:
                s = (time.time() - t0) / args.bench
                print(f"{s * 1000:.0f} ms por lote -> ~{s * len(Xtr) / args.batch / 60:.1f} min por época, ~{s * len(Xtr) / args.batch * args.epochs / 60:.0f} min en total")
                return
        sched.step()
        vp = val_psnr(net, Xva, Yva)
        log.append({"epoch": epoch, "train_mse": round(total / n, 6), "val_psnr": round(vp, 3), "seconds": round(time.time() - t0, 1)})
        mark = ""
        if vp > best:
            best, mark = vp, "  <- mejor"
            torch.save({"state_dict": net.state_dict(), "width": args.width, "epoch": epoch, "val_psnr": vp, "val_base": base}, MODEL)
        print(f"Época {epoch:3d} | MSE {total / n:.5f} | val {vp:.2f} dB | {log[-1]['seconds']} s{mark}", flush=True)
        with open(OUT / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=log[0].keys(), lineterminator="\n")
            w.writeheader()
            w.writerows(log)
    print(f"Mejor PSNR de validación: {best:.2f} dB (sin retocar: {base:.2f}) -> {MODEL.relative_to(ROOT)}")


def evaluate():
    man = json.loads((ROOT / "data" / "fivek_splits.json").read_text("utf-8"))
    find = lambda d, n: next(p for p in (ROOT / d).glob(n + ".*"))
    net, diag = R.load(MODEL), diagnosis.Diagnoser(ROOT / "models" / "diagnosis_mlp.pt")
    rows = {"entrada": [], "reglas_actuales": [], "reglas_y_balance": [], "retoque_aprendido": []}
    t0 = time.time()
    for n in man["splits"]["test"]:
        x = cv2.imread(str(find(man["input_dir"], n))).astype(np.float32) / 255
        y = cv2.imread(str(find(man["expert_dir"], n))).astype(np.float32) / 255
        tone = [s for s in pipeline.plan(diag(x)["problemas"]) if s in ("contraste", "luz")]
        rules = pipeline.run(x, tone, {})[0]
        outs = {"entrada": x, "reglas_actuales": rules, "reglas_y_balance": enhance.white_balance(rules)[0],
                "retoque_aprendido": R.retouch(net, x)[0]}
        for k, o in outs.items():
            rows[k].append((metrics.psnr(y, o), metrics.ssim(y, o)))
    res = {k: {"psnr": float(np.mean([r[0] for r in v])), "ssim": float(np.mean([r[1] for r in v])),
               "mejora_vs_entrada": f"{sum(r[0] > e[0] + 0.05 for r, e in zip(v, rows['entrada']))}/{len(v)}"} for k, v in rows.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test.json").write_text(json.dumps({"fotos": len(rows["entrada"]), "resultados": res}, indent=1, ensure_ascii=False), "utf-8")
    lines = ["| Método | PSNR (dB) | SSIM | Mejora frente a la entrada |", "|---|---|---|---|"]
    lines += [f"| {k} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['mejora_vs_entrada']} |" for k, r in res.items()]
    (OUT / "test.md").write_text(f"FiveK, {len(rows['entrada'])} fotos de prueba completas (480p), contra el retoque del experto C.\n\n"
                                 + "\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines) + f"\n{time.time() - t0:.0f} s -> results/retouch_fivek/test.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if not args.eval_only:
        train(args)
    if not args.bench:
        evaluate()


if __name__ == "__main__":
    main()
