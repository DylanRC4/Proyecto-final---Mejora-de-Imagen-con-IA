"""Grafica la evolución del entrenamiento a partir de results/<name>/train_log.csv.

Uso:  python scripts/plot_training.py [--name denoise_cnn]
Salida: results/<name>/curvas.png (MSE de entrenamiento y PSNR/SSIM de validación por época).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="denoise_cnn")
    d = ROOT / "results" / ap.parse_args().name
    rows = list(csv.DictReader(open(d / "train_log.csv", encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    env = json.loads((d / "config.json").read_text("utf-8"))["env"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(ep, [float(r["train_mse"]) for r in rows], "o-", ms=3)
    a1.set(title="Pérdida de entrenamiento (MSE)", xlabel="Época", ylabel="MSE")
    a1.grid(alpha=0.3)
    a2.plot(ep, [float(r["val_psnr"]) for r in rows], "o-", ms=3, label="CNN")
    a2.axhline(env["val_noisy_psnr"], ls="--", c="gray", label="Sin filtrar")
    a2.set(title="PSNR de validación (σ=25)", xlabel="Época", ylabel="dB")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "curvas.png", dpi=120)
    print("->", d / "curvas.png")


if __name__ == "__main__":
    main()
