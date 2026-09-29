"""Modo automático: con el diagnóstico decide qué mejoras aplicar y en qué ORDEN.

El orden importa: primero se quita el ruido, porque aclarar, contrastar o dar nitidez lo
amplifican; después se recupera el detalle; al final se corrigen contraste, luz y color, que
son ajustes globales. Cada paso registra de dónde viene (nuestra IA, clásico o externo).
Tono: primero niveles (punto negro y blanco), porque subexponer una foto la escala casi
linealmente y los niveles lo deshacen sin crear el "velo gris" que deja la gamma sobre negros
levantados; después se VUELVE A MEDIR la foto y la gamma y el color solo actúan si el problema sigue.
El modo automático solo usa nuestras redes y métodos clásicos: nada generativo. Real-ESRGAN
(externo) queda como herramienta manual, en los pasos terminados en "_externo".
"""
import time

import cv2
import numpy as np

from . import diagnosis, enhance, external
from .io_utils import resize_max
from .model import denoise as cnn  # aplica cualquiera de nuestras CNN residuales, por bloques

ORIGEN = {"ruido": "Nuestra CNN de ruido (entrenada por nosotros)",
          "detalle": "Nuestra CNN de detalle (entrenada por nosotros)",
          "detalle_externo": "Real-ESRGAN (externo, preentrenado, generativo)",
          "luz": "Clásico: corrección gamma", "contraste": "Clásico: niveles automáticos",
          "color": "Clásico: balance de blancos gray-edge", "nitidez": "Clásico: máscara de desenfoque",
          "ampliar": "Bicúbica ×2 + nuestra CNN de detalle",
          "ampliar_externo": "Real-ESRGAN (externo, preentrenado, generativo)"}
ORDEN = ["ruido", "detalle", "detalle_externo", "contraste", "luz", "color", "nitidez", "ampliar", "ampliar_externo"]
PROPIOS = [p for p in ORDEN if not p.endswith("_externo")]
# CNN de detalle que usa la app. v2 elegida en las 200 fotos de prueba (results/detail_eval): supera a la v1 en
# desenfoque (30.59 vs 28.22 dB), JPEG (mejora 163/200 fotos vs 34/200) y baja resolución; la v1 solo gana en "fija".
DETAIL_MODEL = "detail_cnn_v2.pt"


def plan(problemas: dict) -> list[str]:
    """Traduce el diagnóstico en pasos, siempre en el orden de ORDEN."""
    luz = problemas["oscura"] or problemas["sobreexpuesta"]
    quiere = {"ruido": problemas["ruido"], "detalle": problemas["desenfoque"] or problemas["compresion"],
              "contraste": problemas["poco_contraste"] or luz, "luz": luz, "color": problemas["dominante_color"]}
    return [p for p in ORDEN if quiere.get(p)]


def apply_step(step: str, img: np.ndarray, models: dict, factor: int = 2, amount: float = 0.6,
               strength: float = 1.0) -> tuple[np.ndarray, str]:
    esrgan, detail = models.get("esrgan"), models.get("detail")
    if step.endswith("_externo") and esrgan is None:
        raise RuntimeError("Real-ESRGAN no está instalado: python scripts/download_models.py")
    if step == "ruido":
        return cnn(models["denoiser"], img), "ruido estimado y restado"
    if step == "detalle":
        if detail is None:
            return enhance.unsharp(img, amount), "falta la CNN de detalle: se usó máscara de desenfoque"
        return cnn(detail, img), "desenfoque y compresión corregidos (entrenada con MSE: no inventa texturas)"
    if step == "detalle_externo":
        # Mezcla lineal con la entrada: strength=1 es Real-ESRGAN puro; menos suaviza su aspecto "pintado".
        out = strength * external.restore(esrgan, img) + (1 - strength) * img
        return out.astype(np.float32), f"intensidad {strength:.0%}"
    if step in ("luz", "color"):  # se vuelve a medir con las MISMAS reglas del diagnóstico
        f = diagnosis.features(img)
        t = diagnosis.tone_flags(f)
        if step == "luz" and diagnosis.BRILLO_MIN <= f[5] <= diagnosis.BRILLO_MAX:
            return img, f"no hizo falta: tras los niveles el brillo medio ya es {f[5]:.2f}"
        if step == "color" and not t["dominante_color"]:
            return img, "no hizo falta: tras los pasos anteriores ya no se mide dominante de color"
    if step == "luz":
        out, g = enhance.auto_gamma(img)
        return out, f"γ = {g:.2f} ({'aclara' if g < 1 else 'oscurece'})"
    if step == "contraste":
        out, a, b = enhance.auto_levels(img)
        return out, f"α = {a:.2f}, β = {b * 255:.0f}"
    if step == "color":
        out, gains = enhance.white_balance(img)
        return out, "ganancias B, G, R = " + ", ".join(f"{x:.2f}" for x in gains)
    if step == "nitidez":
        return enhance.unsharp(img, amount), f"intensidad {amount}"
    if step.startswith("ampliar"):
        factor = factor if step == "ampliar_externo" else 2  # la nuestra se entrenó hasta ×3: solo ofrece ×2
        small = resize_max(img, {2: 1024, 4: 512}[factor])  # límite para que el portátil no tarde minutos
        note = f" (entrada reducida a {small.shape[1]}×{small.shape[0]} px)" if small.shape != img.shape else ""
        if step == "ampliar_externo":
            return external.upscale(esrgan, small, factor), f"×{factor}{note}"
        if detail is None:
            raise RuntimeError("Falta la CNN de detalle en models/")
        h, w = small.shape[:2]
        big = np.clip(cv2.resize(small, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC), 0, 1)
        return cnn(detail, big), f"×2{note}: la CNN recupera lo que la bicúbica deja borroso (caso 'baja_res' del entrenamiento)"
    raise ValueError(f"Paso desconocido: {step}")


def run(img: np.ndarray, steps: list[str], models: dict, **opts) -> tuple[np.ndarray, list[dict]]:
    log = []
    for step in sorted(steps, key=ORDEN.index):
        t0 = time.perf_counter()
        img, detalle = apply_step(step, img, models, **opts)
        log.append({"paso": step, "origen": ORIGEN[step], "detalle": detalle, "segundos": round(time.perf_counter() - t0, 2)})
    return img, log
