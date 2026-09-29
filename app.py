"""Revela — interfaz.   Ejecutar:  streamlit run app.py

Todo se procesa en float32 [0, 1] y en orden BGR (como carga OpenCV y como se entrenaron los
modelos); solo se convierte a RGB para mostrar (Sesión 02). Los estilos están en imageenhance/ui.py.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from imageenhance import analysis, classic, diagnosis, enhance, metrics, noise, pipeline, ui
from imageenhance.io_utils import bgr_to_rgb, decode_bytes, resize_max, to_float, to_uint8

ROOT = Path(__file__).resolve().parent
EXAMPLE_DIRS = {"demo": ROOT / "data" / "demo", "bsds": ROOT / "data" / "raw" / "bsds500" / "test"}
NOMBRES = {"ruido": "Ruido", "detalle": "Detalle (desenfoque / compresión)", "luz": "Iluminación",
           "contraste": "Contraste", "retoque": "Retoque aprendido (FiveK)", "color": "Color", "nitidez": "Nitidez extra", "ampliar": "Ampliar ×2",
           "detalle_externo": "Detalle con Real-ESRGAN (externo)", "ampliar_externo": "Ampliar con Real-ESRGAN (externo)"}
PROBLEMAS = {"ruido": "Ruido", "desenfoque": "Desenfoque", "compresion": "Compresión JPEG"}

st.set_page_config(page_title="Revela", layout="wide")
ui.style()


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models():
    from imageenhance import external, model
    m = {"denoiser": None, "detail": None, "diagnoser": None, "esrgan": None, "retouch": None, "meta": {}}
    if (ROOT / "models" / pipeline.NOISE_MODEL).exists():
        m["denoiser"], m["meta"] = model.load(ROOT / "models" / pipeline.NOISE_MODEL)
    if (ROOT / "models" / pipeline.DETAIL_MODEL).exists():
        m["detail"] = model.load(ROOT / "models" / pipeline.DETAIL_MODEL)[0]
    if (ROOT / "models" / "diagnosis_mlp.pt").exists():
        m["diagnoser"] = diagnosis.Diagnoser(ROOT / "models" / "diagnosis_mlp.pt")
    if (ROOT / "models" / "retouch_fivek.pt").exists():
        from imageenhance import retouch
        m["retouch"] = retouch.load(ROOT / "models" / "retouch_fivek.pt")
    if external.ESRGAN_PATH.exists():
        m["esrgan"] = external.load_esrgan()
    return m


@st.cache_data(show_spinner="Procesando…", max_entries=8)
def run_steps(img: np.ndarray, steps: tuple, factor: int = 2, amount: float = 0.6, strength: float = 0.7):
    return pipeline.run(img, list(steps), load_models(), factor=factor, amount=amount, strength=strength)


@st.cache_data(show_spinner="Leyendo la imagen…", max_entries=4)
def prepare(raw: bytes, max_side: int) -> np.ndarray:
    """Decodifica y reduce UNA vez; sin caché, una foto de 74 MB se decodificaría en cada clic."""
    return to_float(resize_max(decode_bytes(raw), max_side))


def rgb(img):
    return bgr_to_rgb(to_uint8(img))


def frame(img):
    """Dónde mostrar la foto: las verticales usan una columna más angosta para caber en la pantalla."""
    r = max(0.35, 0.85 * img.shape[1] / img.shape[0])
    return st.container() if r >= 0.85 else st.columns([r, 1 - r])[0]


def compare(a, b, key):
    """Antes/después en tres vistas: deslizador (misma foto partida), lado a lado y lupa ×3."""
    same = a.shape == b.shape
    view = st.radio("Vista", ["Deslizador", "Lado a lado", "Lupa ×3"] if same else ["Lado a lado"], horizontal=True,
                    key=f"vista_{key}", label_visibility="collapsed")
    if view == "Deslizador":
        box = frame(a)
        box.image(ui.split(rgb(a), rgb(b), st.session_state.get(f"corte_{key}", 50)), width="stretch")
        box.slider("Corte", 0, 100, 50, key=f"corte_{key}", label_visibility="collapsed")
        ui.html('<div class="rv-legend"><span>← Original</span><span>Resultado →</span></div>', box)
    elif view == "Lado a lado":
        c1, c2 = st.columns(2)
        c1.image(rgb(a), caption=f"Original — {a.shape[1]}×{a.shape[0]} px", width="stretch")
        c2.image(rgb(b), caption=f"Resultado — {b.shape[1]}×{b.shape[0]} px", width="stretch")
    else:
        zoom(a, b)


def zoom(a, b, side=200):
    """Lupa: la zona donde MÁS cambió la foto, ampliada ×3 sin suavizar (píxeles reales). A pantalla
    completa la foto se ve reducida y un cambio de 1-2 px (grano, bordes) no se nota."""
    s = min(side, *a.shape[:2])
    y, x = np.unravel_index(np.argmax(cv2.boxFilter(np.abs(b - a).mean(axis=2), -1, (s, s))), a.shape[:2])
    y, x = min(max(y - s // 2, 0), a.shape[0] - s), min(max(x - s // 2, 0), a.shape[1] - s)
    big = lambda im: cv2.resize(rgb(im[y:y + s, x:x + s]), (3 * s, 3 * s), interpolation=cv2.INTER_NEAREST)
    c1, c2 = st.columns(2)
    c1.image(big(a), caption=f"Antes — zona x={x}, y={y}", width="stretch")
    c2.image(big(b), caption="Después — la zona donde más cambió la foto", width="stretch")


def reference_metrics(ref, before, after):
    if ref is None:
        ui.note("Foto real: no existe la versión limpia, así que PSNR y SSIM no se pueden calcular. Compara a la vista.")
        return
    if after.shape != ref.shape:
        ui.note("La imagen cambió de tamaño: PSNR y SSIM solo se comparan a la misma resolución.")
        return
    e0, e1 = metrics.evaluate(ref, before), metrics.evaluate(ref, after)
    st.metric("PSNR", f"{e1['psnr']:.2f} dB", f"{e1['psnr'] - e0['psnr']:+.2f} dB frente a la entrada")
    st.metric("SSIM", f"{e1['ssim']:.3f}", f"{e1['ssim'] - e0['ssim']:+.3f} frente a la entrada")
    with st.expander("Tabla de métricas"):
        st.dataframe(pd.DataFrame({"Entrada degradada": e0, "Resultado": e1}).T
                     .rename(columns={"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑"}).round(3), width="stretch")
        st.caption("Válido porque la degradación la aplicamos nosotros sobre una foto limpia que conocemos.")


def report(log, where):
    """Pasos aplicados (línea de tiempo) y, debajo, el registro completo como tabla."""
    with where:
        ui.section("Qué hizo el sistema", "Cada paso indica si viene de nuestra IA, de un método clásico o de un modelo externo.")
        ui.steps_log(log, NOMBRES)
        with st.expander("Registro completo"):
            st.dataframe(pd.DataFrame(log).set_index("paso").rename(index=NOMBRES), width="stretch")


def download(img, name, key):
    ok, buf = cv2.imencode(".png", to_uint8(img))
    st.download_button("Descargar PNG", buf.tobytes(), file_name=f"{Path(name).stem}_revelada.png", mime="image/png",
                       key=key, width="stretch")


models = load_models()

# ---------------- Barra lateral ----------------
ui.sidebar_brand()
ui.label("Fotografía")
up = st.sidebar.file_uploader("Cargar fotografía", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                              label_visibility="collapsed")
examples = [f"{k}/{p.name}" for k, d in EXAMPLE_DIRS.items() if d.exists()
            for p in sorted(d.iterdir())[: (40 if k == "demo" else 20)] if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
if up is not None:
    raw, src_name = up.getvalue(), up.name
elif examples:
    choice = st.sidebar.selectbox("O una foto de ejemplo", examples, key="ejemplo")
    raw, src_name = (EXAMPLE_DIRS[choice.split("/")[0]] / choice.split("/", 1)[1]).read_bytes(), choice
else:
    ui.header("Sin fotografía", 0, 0, "—")
    ui.note("Carga una fotografía en la barra lateral para comenzar.")
    st.stop()

ui.label("Modo")
mode = st.sidebar.radio("Modo", ["Foto real (sin referencia)", "Experimento: degradar una foto limpia"], key="modo",
                        label_visibility="collapsed")
experiment = mode.startswith("Experimento")
if experiment:
    kind = st.sidebar.selectbox("Degradación", ["gaussian", "salt_pepper", "blur", "jpeg"], key="degradacion",
                                format_func={"gaussian": "Ruido gaussiano", "salt_pepper": "Sal y pimienta",
                                             "blur": "Desenfoque", "jpeg": "Compresión JPEG"}.get)
    level = {"gaussian": lambda: st.sidebar.slider("Sigma del ruido (0-255)", 5, 50, 25),
             "salt_pepper": lambda: st.sidebar.slider("Fracción de píxeles", 0.01, 0.20, 0.05, 0.01),
             "blur": lambda: st.sidebar.slider("Sigma del desenfoque", 0.5, 3.0, 1.5, 0.1),
             "jpeg": lambda: st.sidebar.slider("Calidad JPEG (menor = peor)", 5, 60, 15)}[kind]()
    seed = int(st.sidebar.number_input("Semilla", 0, 10_000, 42))
ui.label("Procesamiento")
max_side = st.sidebar.slider("Lado máximo (px)", 256, 2048, 1280, 64,
                             help="Todo corre en CPU: fotos más pequeñas se procesan más rápido.")

ui.label("Modelos")
ui.models_list([(n, models[k] is not None, o) for n, k, o in [
    ("CNN de ruido", "denoiser", "propia"), ("CNN de detalle", "detail", "propia"), ("CNN de retoque (FiveK)", "retouch", "propia"),
    ("MLP de diagnóstico", "diagnoser", "propia"), ("Real-ESRGAN", "esrgan", "externa")]])
if models["esrgan"] is None:
    st.sidebar.caption("Real-ESRGAN es opcional (solo ajuste manual): `python scripts/download_models.py`")
ui.team(["Dylan Ricaurte", "Brayan Garcia", "Nicolas Fontecha", "Esteban Monroy"])

original = prepare(raw, max_side)
reference, work = (original, noise.degrade(original, kind, level, seed)) if experiment else (None, original)
img_key = hashlib.md5(work.tobytes()).hexdigest()

ui.header(src_name, work.shape[1], work.shape[0], "Experimento con referencia" if experiment else "Foto real")
tab_auto, tab_tools, tab_an, tab_tec = st.tabs(["Restauración automática", "Ajuste manual", "Análisis", "Filtros vs. CNN"])

# ---------------- 1. Restauración automática ----------------
with tab_auto:
    if models["diagnoser"] is None or models["denoiser"] is None:
        st.warning("Faltan modelos entrenados en models/. Ejecuta scripts/train.py y scripts/train_diagnosis.py.")
    else:
        diag = models["diagnoser"](work)  # se diagnostica a la misma resolución que se va a procesar
        probs, prob, f = diag["problemas"], diag["prob"], diag["features"]
        left, right = st.columns([2.3, 1], gap="large")
        with right:
            ui.section("1 · Diagnóstico", "Probabilidades de nuestro MLP y reglas del histograma.")
            ui.bars(prob, probs, PROBLEMAS)
            ui.flags({"Oscura": probs["oscura"], "Sobreexpuesta": probs["sobreexpuesta"],
                      "Poco contraste": probs["poco_contraste"], "Dominante de color": probs["dominante_color"]})
            with st.expander("Valores del diagnóstico"):
                st.dataframe(pd.DataFrame({"probabilidad": [f"{prob[k] * 100:.0f} %" for k in prob],
                                           "detectado": ["sí" if probs[k] else "no" for k in prob]}, index=list(prob)), width="stretch")
                st.dataframe(pd.DataFrame({"medida": [f"p99 {f['percentil_99']:.2f} (mín. {diagnosis.P99_MIN})",
                                                      f"p1 {f['percentil_1']:.2f} (máx. {diagnosis.P1_MAX})",
                                                      f"rango {f['percentil_99'] - f['percentil_1']:.2f}",
                                                      f"bordes {f['dominante_bordes']:.2f} (umbral {diagnosis.COLOR_UMBRAL})"],
                                           "detectado": ["sí" if probs[k] else "no" for k in
                                                         ("oscura", "sobreexpuesta", "poco_contraste", "dominante_color")]},
                                          index=["oscura", "sobreexpuesta", "poco contraste", "dominante de color"]), width="stretch")
            # Retoque aprendido: estético (se compara contra un fotógrafo, no contra la foto limpia), así que en el modo
            # experimento arranca apagado para que el PSNR mida solo la restauración.
            retoque = models["retouch"] is not None and st.checkbox(
                "Retoque aprendido (FiveK)", value=not experiment, key=f"retoque_{experiment}",
                help="Nuestra CNN ajusta luz, contraste y color como lo haría el fotógrafo C de MIT-Adobe FiveK.")
            plan = pipeline.plan(probs, retoque=retoque)
            if probs["dominante_color"]:  # el color no se corrige solo (ver pipeline.plan): decide quien conoce la escena
                ui.note(f"Posible dominante de color (bordes {f['dominante_bordes']:.2f} > {diagnosis.COLOR_UMBRAL}). No se "
                        "corrige sola: puede ser la luz de la escena (atardecer, faroles) o un defecto (bombillo amarillo).")
                if st.checkbox("Es un defecto: neutralizar el color", key="neutral", help="Balance de blancos gray-edge (clásico)."):
                    plan.append("color")
            ui.section("2 · Plan de restauración", "Lo decide el diagnóstico; cada paso mide la foto para saber cuánto corregir.")
            if not plan:
                ui.note("El diagnóstico no encontró problemas claros: la foto se deja como está. "
                        "Si quieres otro ajuste, usa la pestaña Ajuste manual.")
            else:
                ui.chips(plan, NOMBRES, pipeline.ORIGEN)
            if plan and st.button("Revelar foto", type="primary", key="auto_btn", width="stretch"):
                st.session_state["auto"] = img_key
        if plan and st.session_state.get("auto") == img_key:
            out, log = run_steps(work, tuple(plan))
            with left:
                compare(work, out, "auto")
            report(log, left)
            with right:
                ui.section("3 · Resultado")
                reference_metrics(reference, work, out)
                download(out, src_name, "dl_auto")
        else:
            with left:
                frame(work).image(rgb(work), caption="Original — pulsa «Revelar foto» para restaurarla", width="stretch")

# ---------------- 2. Ajuste manual ----------------
with tab_tools:
    left, right = st.columns([2.3, 1], gap="large")
    with right:
        ui.section("Herramientas", "Se aplican de arriba hacia abajo: ruido → detalle → contraste → luz → color → nitidez → ampliar.")
        t_noise = st.selectbox("Reducir ruido", ["No", "Nuestra CNN", "Mediana", "Gaussiano"], key="t_ruido")
        t_detail = st.selectbox("Restaurar detalle", ["No", "Nuestra CNN de detalle", "Real-ESRGAN (externo, generativo)"],
                                key="t_detalle")
        t_force = st.slider("Intensidad de Real-ESRGAN", 0.0, 1.0, 0.7, 0.1, key="t_fuerza",
                            help="1 = modelo puro (a veces aspecto 'pintado'); menos = mezcla con la original.") \
            if t_detail.startswith("Real") else 0.7
        t_contrast = st.selectbox("Contraste", ["No", "Niveles automáticos", "CLAHE (por zonas)", "Retoque aprendido (FiveK)"],
                                  key="t_contraste")
        t_light = st.selectbox("Iluminación", ["No", "Automática (gamma)", "Manual"], key="t_luz")
        gamma = st.slider("γ (menor que 1 aclara)", 0.3, 3.0, 1.0, 0.05, key="t_gamma") if t_light == "Manual" else None
        t_color = st.slider("Balance de blancos (intensidad)", 0.0, 1.0, 0.0, 0.1, key="t_color")
        t_sharp = st.slider("Nitidez (máscara de desenfoque)", 0.0, 2.0, 0.0, 0.1, key="t_nitidez")
        t_up = st.selectbox("Ampliar resolución", ["No", "×2 — nuestra CNN", "×2 — Real-ESRGAN (externo)", "×4 — Real-ESRGAN (externo)"],
                            key="t_ampliar")
    out, log = work, []
    if t_noise == "Nuestra CNN" and models["denoiser"] is not None:
        out, lg = run_steps(out, ("ruido",), 2, 0.6); log += lg
    elif t_noise == "Mediana":
        out = classic.median_filter(out, 3); log.append({"paso": "ruido", "origen": "Clásico: mediana 3×3"})
    elif t_noise == "Gaussiano":
        out = classic.gaussian_filter(out, 7, 0.8); log.append({"paso": "ruido", "origen": "Clásico: gaussiano 7×7, σ=0.8"})
    if t_detail != "No":
        out, lg = run_steps(out, ("detalle" if t_detail.startswith("Nuestra") else "detalle_externo",), 2, 0.6, t_force)
        log += lg
    if t_contrast == "Niveles automáticos":
        out = enhance.auto_levels(out)[0]; log.append({"paso": "contraste", "origen": "Clásico: niveles automáticos"})
    elif t_contrast.startswith("Retoque") and models["retouch"] is not None:
        out, lg = run_steps(out, ("retoque",)); log += lg
    elif t_contrast.startswith("CLAHE"):
        out = enhance.clahe(out); log.append({"paso": "contraste", "origen": "Clásico: CLAHE"})
    if t_light == "Automática (gamma)":
        out, g = enhance.auto_gamma(out); log.append({"paso": "luz", "origen": f"Clásico: gamma automática γ={g:.2f}"})
    elif t_light == "Manual":
        out = np.power(out, gamma, dtype=np.float32); log.append({"paso": "luz", "origen": f"Clásico: gamma manual γ={gamma}"})
    if t_color > 0:
        out = enhance.white_balance(out, t_color)[0]; log.append({"paso": "color", "origen": f"Clásico: balance de blancos gray-edge {t_color}"})
    if t_sharp > 0:
        out = enhance.unsharp(out, t_sharp); log.append({"paso": "nitidez", "origen": f"Clásico: máscara de desenfoque {t_sharp}"})
    if t_up != "No":
        out, lg = run_steps(out, ("ampliar_externo" if "Real" in t_up else "ampliar",), int(t_up[1]), 0.6); log += lg
    if log:
        with left:
            compare(work, out, "tools")
        report(log, left)
        with right:
            reference_metrics(reference, work, out)
            download(out, src_name, "dl_tools")
    else:
        with left:
            frame(work).image(rgb(work), caption="Original — elige una herramienta a la derecha", width="stretch")

# ---------------- 3. Análisis ----------------
with tab_an:
    c1, c2 = st.columns([2.3, 1], gap="large")
    with c1:
        frame(work).image(rgb(work), caption="Imagen de entrada", width="stretch")
    with c2:
        ui.section("Ruido estimado", "Sin referencia: Immerkær (1996).")
        st.metric("Sigma de ruido estimado", f"{analysis.estimate_noise_sigma(work):.1f}")
        st.caption("Las texturas finas la inflan y el desenfoque la baja: no reemplaza al PSNR.")
        ui.section("Canales")
        st.dataframe(pd.DataFrame(analysis.channel_stats(work)).T.round(1), width="stretch")
        ui.section("Histograma")
        h = analysis.histograms(work)
        st.line_chart(pd.DataFrame({"R": h["R"], "G": h["G"], "B": h["B"]}), color=["#e45756", "#54a24b", "#4c78a8"], height=200)
    with st.expander("Magnitud del gradiente (Sobel)"):
        g = analysis.gradient_magnitude(work)
        st.image(np.clip(g / (np.percentile(g, 99) + 1e-9), 0, 1), width="stretch")

# ---------------- 4. Filtros clásicos contra nuestra CNN (ruido) ----------------
with tab_tec:
    ui.section("Filtros clásicos contra nuestra CNN", "Parámetros de los filtros elegidos en validación (results/classic_params.json).")
    p = ROOT / "results" / "classic_params.json"
    best = json.loads(p.read_text("utf-8"))["best"] if p.exists() else {}
    key = f"{kind}_{level}" if experiment else "gaussian_25"
    med = classic.median_filter(work, **best.get(f"{key}_median", {"ksize": 3}))
    gau = classic.gaussian_filter(work, **best.get(f"{key}_gaussian", {"ksize": 7, "sigma": 0.8}))
    results = {"Entrada": work, "Mediana": med, "Gaussiano": gau}
    if models["denoiser"] is not None:
        results["Nuestra CNN"] = run_steps(work, ("ruido",), 2, 0.6)[0]
    for col, (name, im) in zip(st.columns(len(results)), results.items()):
        col.image(rgb(im), caption=name, width="stretch")
    if reference is not None:
        st.dataframe(pd.DataFrame({n: metrics.evaluate(reference, im) for n, im in results.items()}).T
                     .rename(columns={"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑"}).round(3), width="stretch")
    else:
        st.dataframe(pd.DataFrame({n: {"sigma estimado": analysis.estimate_noise_sigma(im)} for n, im in results.items()}).T.round(2))
        st.caption("Ojo: un sigma estimado menor NO significa mejor calidad; el desenfoque también lo baja. Compara a la vista.")
    if "Nuestra CNN" in results:
        ui.section("Dentro de la CNN")
        c1, c2 = st.columns(2)
        c1.image(np.clip(0.5 + 3 * (work - results["Nuestra CNN"]), 0, 1)[..., ::-1],
                 caption="Lo que la CNN estimó como ruido (entrada − salida, ×3)", width="stretch")
        w = models["denoiser"].noise_net[0].weight.detach().numpy()[:, 1]
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)
        tiles = [cv2.resize(k, (48, 48), interpolation=cv2.INTER_NEAREST) for k in w]
        c2.image(np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]), clamp=True, width=380,
                 caption=f"32 filtros 3×3 aprendidos por la 1.ª capa (canal G) — época {models['meta'].get('epoch')}")
