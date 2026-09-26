"""Modo automático: con el diagnóstico decide qué mejoras aplicar y en qué ORDEN.

El orden importa: primero se quita el ruido, porque aclarar, contrastar o dar nitidez lo
amplifican; después se recupera el detalle; al final se corrigen luz, contraste y color, que
son ajustes globales. Cada paso registra de dónde viene (nuestra IA, clásico o externo).
"""
import time

import numpy as np

from . import enhance, external
from .io_utils import resize_max
from .model import denoise

ORIGEN = {"ruido": "Nuestra CNN (entrenada por nosotros)",
          "detalle": "Real-ESRGAN (modelo externo preentrenado)",
          "luz": "Clásico: corrección gamma", "contraste": "Clásico: niveles automáticos",
          "color": "Clásico: balance de blancos", "nitidez": "Clásico: máscara de desenfoque",
          "ampliar": "Real-ESRGAN (modelo externo preentrenado)"}
ORDEN = ["ruido", "detalle", "luz", "contraste", "color", "nitidez", "ampliar"]


def plan(problemas: dict) -> list[str]:
    """Traduce el diagnóstico en pasos, siempre en el orden de ORDEN."""
    quiere = {"ruido": problemas["ruido"], "detalle": problemas["desenfoque"] or problemas["compresion"],
              "luz": problemas["oscura"] or problemas["sobreexpuesta"], "contraste": problemas["poco_contraste"],
              "color": problemas["dominante_color"]}
    return [p for p in ORDEN if quiere.get(p)]


def apply_step(step: str, img: np.ndarray, models: dict, factor: int = 2, amount: float = 0.6,
               strength: float = 1.0) -> tuple[np.ndarray, str]:
    esrgan = models.get("esrgan")
    if step == "ruido":
        return denoise(models["denoiser"], img), "ruido estimado y restado"
    if step == "detalle":
        if esrgan is None:
            return enhance.unsharp(img, amount), "Real-ESRGAN no disponible: se usó máscara de desenfoque"
        # Mezcla lineal con la entrada: strength=1 es Real-ESRGAN puro; menos suaviza su aspecto "pintado".
        out = strength * external.restore(esrgan, img) + (1 - strength) * img
        return out.astype(np.float32), f"nitidez y compresión restauradas (intensidad {strength:.0%})"
    if step == "luz":
        out, g = enhance.auto_gamma(img)
        return out, f"γ = {g:.2f} ({'aclara' if g < 1 else 'oscurece'})"
    if step == "contraste":
        out, a, b = enhance.auto_levels(img)
        return out, f"α = {a:.2f}, β = {b * 255:.0f}"
    if step == "color":
        out, gains = enhance.white_balance(img, strength=0.5)
        return out, "ganancias B, G, R = " + ", ".join(f"{x:.2f}" for x in gains)
    if step == "nitidez":
        return enhance.unsharp(img, amount), f"intensidad {amount}"
    if step == "ampliar":
        if esrgan is None:
            raise RuntimeError("Para ampliar se necesita Real-ESRGAN: python scripts/download_models.py")
        max_in = {2: 1024, 4: 512}[factor]  # límite para que el portátil no tarde minutos
        small = resize_max(img, max_in)
        note = f" (entrada reducida a {small.shape[1]}×{small.shape[0]} px)" if small.shape != img.shape else ""
        return external.upscale(esrgan, small, factor), f"×{factor}{note}"
    raise ValueError(f"Paso desconocido: {step}")


def run(img: np.ndarray, steps: list[str], models: dict, **opts) -> tuple[np.ndarray, list[dict]]:
    log = []
    for step in sorted(steps, key=ORDEN.index):
        t0 = time.perf_counter()
        img, detalle = apply_step(step, img, models, **opts)
        log.append({"paso": step, "origen": ORIGEN[step], "detalle": detalle, "segundos": round(time.perf_counter() - t0, 2)})
    return img, log
