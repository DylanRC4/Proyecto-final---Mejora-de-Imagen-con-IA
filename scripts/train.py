"""Entrenamiento supervisado de la CNN con pares (limpia, ruidosa) generados al vuelo.

En cada época se recortan `patches_per_image` parches aleatorios de CADA foto de entrenamiento
(con giros/espejos) y se les suma ruido gaussiano con sigma aleatorio en [sigma_min, sigma_max].
Todo sale de un generador con semilla (seed, época): el entrenamiento es reproducible.
Validación: recorte central de cada foto de validación con ruido fijo (sigma = val_sigma).
Se guarda el modelo de la época con mejor PSNR de validación. Las fotos de prueba no se tocan.

Uso:
  python scripts/train.py --bench 30   # mide 30 lotes y estima el tiempo total, sin guardar nada
  python scripts/train.py              # entrenamiento completo con configs/train.json
  python scripts/train.py --resume     # continúa desde checkpoints/<name>_last.pt
Salidas: models/<name>.pt, results/<name>/train_log.csv y config.json, checkpoints/<name>_last.pt
"""
import argparse
import csv
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import data, metrics, model as M  # noqa: E402


def to_nchw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))


def make_epoch(train_imgs, cfg, epoch):
    rng = np.random.default_rng([cfg["seed"], epoch])
    clean = np.concatenate([data.random_patches(im, cfg["patches_per_image"], cfg["patch"], rng) for _, im in train_imgs])
    clean = clean[rng.permutation(len(clean))].astype(np.float32) / 255.0
    sigma = rng.uniform(cfg["sigma_min"], cfg["sigma_max"], (len(clean), 1, 1, 1)).astype(np.float32) / 255.0
    noisy = np.clip(clean + rng.standard_normal(clean.shape, dtype=np.float32) * sigma, 0.0, 1.0)
    return to_nchw(clean), to_nchw(noisy)


def build_val(cfg, limit=None):
    c, s = cfg["val_crop"], cfg["val_sigma"]
    clean, noisy = [], []
    for name, im in data.load_split("val", limit):
        y, x = (im.shape[0] - c) // 2, (im.shape[1] - c) // 2
        crop = im[y:y + c, x:x + c]
        rng = np.random.default_rng(data.noise_seed(name, "gaussian", s))
        clean.append(crop)
        noisy.append(np.clip(crop + rng.normal(0, s / 255.0, crop.shape).astype(np.float32), 0, 1))
    return np.stack(clean), np.stack(noisy)


@torch.no_grad()
def validate(net, val, device):
    net.eval()
    clean, noisy = val
    out = torch.cat([net(to_nchw(noisy[i:i + 25]).to(device)).cpu() for i in range(0, len(noisy), 25)])
    out = out.clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    ms = [metrics.evaluate(c, o) for c, o in zip(clean, out)]
    return float(np.mean([m["psnr"] for m in ms])), float(np.mean([m["ssim"] for m in ms]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.json")
    ap.add_argument("--name", default="denoise_cnn")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bench", type=int, default=0, help="número de lotes para medir velocidad")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="usar solo N fotos por split (prueba rápida)")
    args = ap.parse_args()

    root = data.ROOT
    cfg = json.loads((root / args.config).read_text("utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_imgs = data.load_split("train", args.limit, as_float=False)
    val = build_val(cfg, args.limit)
    net = M.DenoiseCNN(3, cfg["features"], cfg["depth"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr_min"])
    loss_fn = nn.MSELoss()
    n_patches = len(train_imgs) * cfg["patches_per_image"]
    batches = math.ceil(n_patches / cfg["batch"])
    print(f"Dispositivo: {device} | hilos: {torch.get_num_threads()} | parámetros: {M.count_params(net):,} | "
          f"fotos train/val: {len(train_imgs)}/{len(val[0])} | parches por época: {n_patches} ({batches} lotes)")

    if args.bench:
        clean, noisy = make_epoch(train_imgs, cfg, 1)
        net.train()
        t0 = time.time()
        for i in range(args.bench):
            j = (i * cfg["batch"]) % n_patches
            c, nz = clean[j:j + cfg["batch"]].to(device), noisy[j:j + cfg["batch"]].to(device)
            opt.zero_grad()
            loss_fn(net(nz), c).backward()
            opt.step()
        per_batch = (time.time() - t0) / args.bench
        t0 = time.time()
        validate(net, val, device)
        t_val = time.time() - t0
        epoch_s = per_batch * batches + t_val
        print(f"BENCH: {per_batch * 1000:.0f} ms/lote | validación {t_val:.1f} s | época ≈ {epoch_s / 60:.1f} min | "
              f"{cfg['epochs']} épocas ≈ {epoch_s * cfg['epochs'] / 3600:.2f} h")
        return

    out_dir, ck_dir, model_path = root / "results" / args.name, root / "checkpoints", root / "models" / f"{args.name}.pt"
    for d in (out_dir, ck_dir, model_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    last_path = ck_dir / f"{args.name}_last.pt"
    start, best, log = 1, -math.inf, []
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start, best, log = ck["epoch"] + 1, ck["best"], ck["log"]
        print(f"Reanudando desde la época {start}")

    noisy_psnr, noisy_ssim = np.mean([metrics.psnr(c, n) for c, n in zip(*val)]), np.mean([metrics.ssim(c, n) for c, n in zip(*val)])
    print(f"Validación sin filtrar (sigma={cfg['val_sigma']}): PSNR {noisy_psnr:.2f} dB | SSIM {noisy_ssim:.4f}")
    env = {"python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(),
           "device": device, "platform": platform.platform(), "processor": platform.processor(),
           "dataset_commit": json.loads(data.MANIFEST.read_text("utf-8"))["commit"], "params": M.count_params(net),
           "train_images": len(train_imgs), "val_images": len(val[0]),
           "val_noisy_psnr": round(float(noisy_psnr), 4), "val_noisy_ssim": round(float(noisy_ssim), 4)}
    t_start = time.time()
    for epoch in range(start, cfg["epochs"] + 1):
        t0 = time.time()
        net.train()
        clean, noisy = make_epoch(train_imgs, cfg, epoch)
        total = 0.0
        for i in range(0, n_patches, cfg["batch"]):
            c, nz = clean[i:i + cfg["batch"]].to(device), noisy[i:i + cfg["batch"]].to(device)
            opt.zero_grad()
            loss = loss_fn(net(nz), c)
            loss.backward()
            opt.step()
            total += loss.item() * len(c)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        vp, vs = validate(net, val, device)
        row = {"epoch": epoch, "train_mse": round(total / n_patches, 7), "val_psnr": round(vp, 4),
               "val_ssim": round(vs, 4), "lr": lr, "seconds": round(time.time() - t0, 1)}
        log.append(row)
        mark = ""
        if vp > best:
            best, mark = vp, "  <- mejor"
            M.save(net.cpu(), model_path, epoch=epoch, val_psnr=vp, val_ssim=vs, config=cfg)
            net.to(device)
        print(f"Época {epoch:3d}/{cfg['epochs']} | MSE {row['train_mse']:.6f} | val PSNR {vp:.2f} dB | "
              f"SSIM {vs:.4f} | lr {lr:.2e} | {row['seconds']:.0f} s{mark}", flush=True)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "log": log}, last_path)
        with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys(), lineterminator="\n")
            w.writeheader()
            w.writerows(log)
        env.update(finished_epochs=epoch, best_val_psnr=round(best, 4), updated=datetime.now().isoformat(timespec="seconds"),
                   session_minutes=round((time.time() - t_start) / 60, 1))
        (out_dir / "config.json").write_text(json.dumps({"config": cfg, "env": env}, indent=1), "utf-8")
    print(f"Listo. Mejor PSNR de validación: {best:.2f} dB -> {model_path}")


if __name__ == "__main__":
    main()
