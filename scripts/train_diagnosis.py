"""Entrena el clasificador MLP del modo automático: ¿la foto tiene ruido, desenfoque o compresión?

Datos: cada foto de BSDS500 se degrada K veces con una combinación aleatoria (semilla fija), en el
orden de una cámara (lente → exposición → sensor → compresión):
  desenfoque (p=0.5, sigma 1-4) → exposición (p=0.5, −2 a +1 EV) y dominante de color (p=0.3, ±10-25 %),
  ambas sin etiqueta → ruido (p=0.5: mitad gaussiano sigma 8-40, mitad ruido de CÁMARA a 5e-4 a 1e-2)
  → JPEG (p=0.5, calidad 8-50; si no, p=0.5 de JPEG bueno 75-95, que NO cuenta como problema).
La etiqueta es qué se aplicó. Exposición, dominante y JPEG bueno son "distractores": la v1 del MLP no
los veía; en la evaluación del sistema no detectaba el ruido en fotos oscuras con JPEG 50 y confundía
una dominante de color con ruido en 49 de 200 fotos.
Se respeta el split por foto: train 200×8, val 100×4, test 200×4 muestras.

Se compara contra una línea base simple: el mejor umbral sobre UNA sola característica por
etiqueta (elegido en train). Si el MLP no la supera, no vale la pena usarlo.

Uso:  python scripts/train_diagnosis.py [--limit N]
Salidas: models/diagnosis_mlp.pt, results/diagnosis/metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageenhance import data, diagnosis as D, noise  # noqa: E402

K = {"train": 8, "val": 4, "test": 4}
SEED = 2026


def make_split(split, limit):
    X, Y = [], []
    for i, (name, img) in enumerate(data.load_split(split, limit)):
        for k in range(K[split]):
            rng = np.random.default_rng([SEED, data.SPLITS.index(split), i, k])
            y = rng.random(3) < 0.5  # ruido, desenfoque, compresión
            x = img
            if y[1]:
                x = noise.add_blur(x, rng.uniform(1.0, 4.0))
            if rng.random() < 0.5:
                x = noise.add_exposure(x, rng.uniform(-2.0, 1.0))
            if rng.random() < 0.3:
                gains = np.ones(3, np.float32)
                a, b = rng.permutation(3)[:2]
                m = rng.uniform(0.10, 0.25)
                gains[a], gains[b] = 1 + m, 1 - m
                x = noise.add_cast(x, gains)
            if y[0] and rng.random() < 0.5:  # ruido de celular: granulado y más fuerte en las sombras
                x = noise.add_camera_noise(x, rng, float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))))
            elif y[0]:
                x = noise.add_gaussian(x, rng.uniform(8, 40), seed=int(rng.integers(1 << 31)))
            if y[2]:
                x = noise.add_jpeg(x, int(rng.integers(8, 51)))
            elif rng.random() < 0.5:
                x = noise.add_jpeg(x, int(rng.integers(75, 96)))
            X.append(D.features(x))
            Y.append(y)
    return np.array(X, np.float32), np.array(Y, np.float32)


def scores(pred, y):
    out = {}
    for j, lab in enumerate(D.LABELS):
        tp = int(((pred[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((pred[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((pred[:, j] == 0) & (y[:, j] == 1)).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[lab] = {"exactitud": round(float((pred[:, j] == y[:, j]).mean()), 4), "precision": round(p, 4),
                    "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}
    out["todas_correctas"] = round(float((pred == y).all(axis=1).mean()), 4)
    return out


def best_stumps(X, Y):
    """Línea base: para cada etiqueta, la característica y el umbral (y sentido) con mejor exactitud en train."""
    stumps = []
    for j in range(Y.shape[1]):
        best = (0, 0, 0.0, 1)
        for f in range(X.shape[1]):
            for t in np.percentile(X[:, f], np.linspace(1, 99, 99)):
                for s in (1, -1):
                    acc = ((s * (X[:, f] - t) > 0) == Y[:, j]).mean()
                    best = max(best, (acc, f, float(t), s))
        stumps.append(best[1:])
    return stumps


def apply_stumps(stumps, X):
    return np.stack([(s * (X[:, f] - t) > 0) for f, t, s in stumps], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    t0 = time.time()
    sets = {s: make_split(s, args.limit) for s in data.SPLITS}
    print(f"Características extraídas en {time.time() - t0:.0f} s: " + ", ".join(f"{s} {len(v[0])}" for s, v in sets.items()))

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = sets["train"], sets["val"], sets["test"]
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6  # normalización con datos de train (Sesión 09)
    norm = lambda X: torch.from_numpy((X - mean) / std)
    torch.manual_seed(SEED)
    net = D.build_mlp()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # sigmoide + entropía cruzada binaria, una por etiqueta
    best = (float("inf"), 0, None)
    for epoch in range(1, args.epochs + 1):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(norm(Xtr)), torch.from_numpy(Ytr))
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = loss_fn(net(norm(Xva)), torch.from_numpy(Yva)).item()
        if vloss < best[0]:
            best = (vloss, epoch, {k: v.clone() for k, v in net.state_dict().items()})
        if epoch % 100 == 0:
            print(f"Época {epoch:4d} | pérdida train {loss.item():.4f} | val {vloss:.4f}")
    net.load_state_dict(best[2])
    with torch.no_grad():
        pred = (torch.sigmoid(net(norm(Xte))) > 0.5).float().numpy()

    stumps = best_stumps(Xtr, Ytr)
    res = {"muestras": {s: len(v[0]) for s, v in sets.items()}, "mejor_epoca": best[1], "val_loss": round(best[0], 4),
           "test_mlp": scores(pred, Yte), "test_umbral_simple": scores(apply_stumps(stumps, Xte), Yte),
           "umbrales_simples": [{"etiqueta": l, "caracteristica": D.FEATURES[f], "umbral": round(t, 4), "sentido": s}
                                for l, (f, t, s) in zip(D.LABELS, stumps)]}
    out = data.ROOT / "results" / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf-8")
    torch.save({"state_dict": net.state_dict(), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "features": D.FEATURES, "labels": D.LABELS, "epoch": best[1]}, data.ROOT / "models" / "diagnosis_mlp.pt")
    print(f"\nMejor época {best[1]} (pérdida val {best[0]:.4f}). Exactitud en PRUEBA ({len(Xte)} muestras):")
    for lab in D.LABELS:
        m, b = res["test_mlp"][lab], res["test_umbral_simple"][lab]
        print(f"  {lab:11s} MLP {m['exactitud']:.3f} (F1 {m['f1']:.3f}) | umbral simple {b['exactitud']:.3f} (F1 {b['f1']:.3f})")
    print(f"  Las 3 correctas a la vez: MLP {res['test_mlp']['todas_correctas']:.3f} | umbral {res['test_umbral_simple']['todas_correctas']:.3f}")
    print(f"Total {time.time() - t0:.0f} s -> models/diagnosis_mlp.pt, results/diagnosis/metrics.json")


if __name__ == "__main__":
    main()
