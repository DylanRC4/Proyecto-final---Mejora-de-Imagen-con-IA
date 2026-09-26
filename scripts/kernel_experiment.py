"""Experimento: filtro FIJO (diseñado a mano) contra filtro APRENDIDO.

Entrenamos una "red" de una sola capa: una convolución 5x5 lineal, sin sesgo ni activación,
sobre la luminancia (Sesión 02) con ruido gaussiano sigma=25. Es exactamente el kernel que
programamos con NumPy en la Sesión 01, pero sus 25 pesos los elige el descenso de gradiente
minimizando el MSE, en vez de elegirlos nosotros.

Preguntas que responde (con números medidos, no supuestos):
  1. ¿Qué forma toma el kernel aprendido? ¿Se parece a un gaussiano? ¿Suma ~1?
  2. ¿Supera al mejor gaussiano 5x5? (El sigma del gaussiano se elige sobre los mismos recortes
     de validación donde se mide: eso le da ventaja al filtro fijo, no al aprendido.)
  3. Un filtro lineal tiene un techo (el mejor promedio ponderado posible). La CNN apila capas
     con ReLU (no lineales, Sesiones 11-12) y por eso puede adaptarse a bordes y texturas.

Uso:  python scripts/kernel_experiment.py [--limit N]
Salidas: results/kernel_experiment.json y results/kernel_experiment.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import classic, data, metrics  # noqa: E402
from imageenhance.io_utils import to_gray  # noqa: E402

SIGMA, K, SEED = 25, 5, 2026


def gray_crops(split, limit, size, rng=None):
    out = []
    for name, im in data.load_split(split, limit):
        g = to_gray(im)
        if rng is None:  # validación: recorte central fijo
            y, x = (g.shape[0] - size) // 2, (g.shape[1] - size) // 2
            out.append((name, g[y:y + size, x:x + size]))
        else:
            out += [(name, p[..., 0]) for p in data.random_patches(g[..., None], 16, size, rng)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train = np.stack([p for _, p in gray_crops("train", args.limit, 64, rng)])[:, None]
    val = gray_crops("val", args.limit, 160)
    val_noisy = [np.clip(c + np.random.default_rng(data.noise_seed(n, "gray_gaussian", SIGMA)).normal(0, SIGMA / 255, c.shape), 0, 1)
                 .astype(np.float32) for n, c in val]

    conv = torch.nn.Conv2d(1, 1, K, padding=K // 2, padding_mode="reflect", bias=False)
    opt = torch.optim.Adam(conv.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, 1e-4)
    clean_t = torch.from_numpy(train)
    for step in range(args.steps):
        idx = torch.from_numpy(rng.integers(0, len(train), 64))
        c = clean_t[idx]
        n = (c + torch.from_numpy(rng.standard_normal(c.shape, dtype=np.float32)) * SIGMA / 255).clamp(0, 1)
        opt.zero_grad()
        loss = torch.mean((conv(n) - c) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    learned = conv.weight.detach().numpy()[0, 0]

    def mean_psnr(fn):
        return float(np.mean([metrics.psnr(c, np.clip(fn(nz), 0, 1)) for (_, c), nz in zip(val, val_noisy)]))

    gauss = {s: mean_psnr(lambda x, s=s: classic.conv2d(x, classic.gaussian_kernel(K, s))) for s in (0.5, 0.7, 0.9, 1.1, 1.4, 1.8)}
    best_s = max(gauss, key=gauss.get)
    res = {"sigma_ruido": SIGMA, "kernel_size": K, "train_patches": len(train), "val_crops": len(val), "steps": args.steps, "nota": "sigma del gaussiano elegido sobre los mismos recortes de validacion (ventaja para el filtro fijo)",
           "psnr_val": {"sin_filtro": mean_psnr(lambda x: x), f"gaussiano_{K}x{K}_sigma_{best_s}": gauss[best_s],
                        f"aprendido_{K}x{K}": mean_psnr(lambda x: classic.conv2d(x, learned))},
           "gaussiano_por_sigma": gauss, "kernel_aprendido": learned.round(4).tolist(),
           "suma_kernel_aprendido": float(learned.sum()), "kernel_gaussiano": classic.gaussian_kernel(K, best_s).round(4).tolist()}
    out = data.ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "kernel_experiment.json").write_text(json.dumps(res, indent=1), "utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(f"Gaussiano fijo σ={best_s}\n(diseñado a mano)", classic.gaussian_kernel(K, best_s)),
              (f"Aprendido por gradiente\n(suma = {learned.sum():.3f})", learned)]
    model_path = data.ROOT / "models" / "denoise_cnn.pt"
    first = None
    if model_path.exists():
        from imageenhance import model as M
        net, _ = M.load(model_path)
        first = net.noise_net[0].weight.detach().numpy()[:, 1]  # canal G de los 32 filtros 3x3
    fig, axes = plt.subplots(1, 3 if first is not None else 2, figsize=(15 if first is not None else 9, 4.6))
    vmax = max(abs(p[1]).max() for p in panels)
    for ax, (title, k) in zip(axes, panels):
        ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for (i, j), v in np.ndenumerate(k):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(title)
        ax.axis("off")
    if first is not None:
        grid = np.ones((4 * 4 - 1, 8 * 4 - 1)) * np.nan
        for n, f in enumerate(first):
            r, c = divmod(n, 8)
            grid[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = f
        m = np.nanmax(abs(first))
        axes[2].imshow(grid, cmap="RdBu_r", vmin=-m, vmax=m)
        axes[2].set_title("CNN: 32 filtros 3×3 de la 1.ª capa (canal G)")
        axes[2].axis("off")
    p = res["psnr_val"]
    fig.suptitle(" | ".join(f"{k}: {v:.2f} dB" for k, v in p.items()) + f"   (validación, ruido σ={SIGMA})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "kernel_experiment.png", dpi=120)
    print(json.dumps(res["psnr_val"], indent=1), "\nsuma del kernel aprendido:", round(float(learned.sum()), 4))
    print(np.array2string(learned, precision=3, suppress_small=True))


if __name__ == "__main__":
    main()
