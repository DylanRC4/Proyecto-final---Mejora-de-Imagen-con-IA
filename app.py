"""ImageEnhance AI — interfaz.   Ejecutar:  streamlit run app.py   (o doble clic en iniciar.bat)

Todo se procesa en float32 [0, 1] y en orden BGR (como carga OpenCV y como se entrenaron los
modelos); solo se convierte a RGB para mostrar (Sesión 02).
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from imageenhance import analysis, classic, diagnosis, enhance, metrics, noise, pipeline
from imageenhance.io_utils import bgr_to_rgb, decode_bytes, resize_max, to_float, to_uint8

ROOT = Path(__file__).resolve().parent
EXAMPLE_DIRS = {"demo": ROOT / "data" / "demo", "bsds": ROOT / "data" / "raw" / "bsds500" / "test"}
NOMBRES = {"ruido": "Ruido", "detalle": "Detalle (desenfoque / compresión)", "luz": "Iluminación",
           "contraste": "Contraste", "color": "Color", "nitidez": "Nitidez extra", "ampliar": "Ampliar ×2",
           "detalle_externo": "Detalle con Real-ESRGAN (externo)", "ampliar_externo": "Ampliar con Real-ESRGAN (externo)"}

st.set_page_config(page_title="ImageEnhance AI", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models():
    from imageenhance import external, model
    m = {"denoiser": None, "detail": None, "diagnoser": None, "esrgan": None, "meta": {}}
    if (ROOT / "models" / "denoise_cnn.pt").exists():
        m["denoiser"], m["meta"] = model.load(ROOT / "models" / "denoise_cnn.pt")
    if (ROOT / "models" / pipeline.DETAIL_MODEL).exists():
        m["detail"] = model.load(ROOT / "models" / pipeline.DETAIL_MODEL)[0]
    if (ROOT / "models" / "diagnosis_mlp.pt").exists():
        m["diagnoser"] = diagnosis.Diagnoser(ROOT / "models" / "diagnosis_mlp.pt")
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


def before_after(a, b, cap_a="Original", cap_b="Mejorada"):
    c1, c2 = st.columns(2)
    c1.image(rgb(a), caption=f"{cap_a} — {a.shape[1]}×{a.shape[0]} px", width="stretch")
    c2.image(rgb(b), caption=f"{cap_b} — {b.shape[1]}×{b.shape[0]} px", width="stretch")


def reference_metrics(ref, before, after):
    if ref is None:
        st.info("Foto real: no existe la versión limpia, así que PSNR y SSIM no se pueden calcular. Compara a la vista.")
        return
    if after.shape != ref.shape:
        st.info("La imagen cambió de tamaño: PSNR y SSIM solo se comparan a la misma resolución.")
        return
    rows = {"Entrada degradada": metrics.evaluate(ref, before), "Resultado": metrics.evaluate(ref, after)}
    st.dataframe(pd.DataFrame(rows).T.rename(columns={"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑"}).round(3), width="stretch")
    st.caption("Válido porque la degradación la aplicamos nosotros sobre una foto limpia que conocemos.")


def download(img, name, key):
    ok, buf = cv2.imencode(".png", to_uint8(img))
    st.download_button("Descargar PNG", buf.tobytes(), file_name=f"{Path(name).stem}_mejorada.png", mime="image/png", key=key)


models = load_models()

# ---------------- Barra lateral ----------------
st.sidebar.title("ImageEnhance AI")
up = st.sidebar.file_uploader("Cargar fotografía", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
examples = [f"{k}/{p.name}" for k, d in EXAMPLE_DIRS.items() if d.exists()
            for p in sorted(d.iterdir())[: (40 if k == "demo" else 20)] if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
if up is not None:
    raw, src_name = up.getvalue(), up.name
elif examples:
    choice = st.sidebar.selectbox("…o una foto de ejemplo", examples, key="ejemplo")
    raw, src_name = (EXAMPLE_DIRS[choice.split("/")[0]] / choice.split("/", 1)[1]).read_bytes(), choice
else:
    st.info("Carga una fotografía en la barra lateral para comenzar.")
    st.stop()

max_side = st.sidebar.slider("Lado máximo para procesar (px)", 256, 2048, 1280, 64,
                             help="Todo corre en CPU: fotos más pequeñas se procesan más rápido.")
mode = st.sidebar.radio("Modo", ["Foto real (sin referencia)", "Experimento: degradar una foto limpia"], key="modo")
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

st.sidebar.markdown("**Modelos**")
for label, key, origin in [("CNN de ruido", "denoiser", "nuestra"), ("CNN de detalle", "detail", "nuestra"),
                           ("MLP de diagnóstico", "diagnoser", "nuestra"), ("Real-ESRGAN", "esrgan", "externo")]:
    st.sidebar.caption(f"{'✅' if models[key] is not None else '❌'} {label} ({origin})")
if models["esrgan"] is None:
    st.sidebar.caption("Real-ESRGAN es opcional (solo herramientas manuales): `python scripts/download_models.py`")

original = prepare(raw, max_side)
reference, work = (original, noise.degrade(original, kind, level, seed)) if experiment else (None, original)
img_key = hashlib.md5(work.tobytes()).hexdigest()

st.title("ImageEnhance AI")
st.caption(f"Mejora de fotografías con IA propia, modelos externos identificados y procesamiento clásico. "
           f"Imagen: {src_name} — {work.shape[1]}×{work.shape[0]} px.")
tab_auto, tab_tools, tab_an, tab_tec = st.tabs(["Mejora automática", "Herramientas", "Análisis", "Comparación técnica"])

# ---------------- 1. Mejora automática ----------------
with tab_auto:
    if models["diagnoser"] is None or models["denoiser"] is None:
        st.warning("Faltan modelos entrenados en models/. Ejecuta scripts/train.py y scripts/train_diagnosis.py.")
    else:
        diag = models["diagnoser"](work)  # se diagnostica a la misma resolución que se va a procesar
        probs, prob = diag["problemas"], diag["prob"]
        st.subheader("1. Diagnóstico")
        c1, c2 = st.columns(2)
        c1.markdown("**Clasificador MLP (nuestra IA)**")
        c1.dataframe(pd.DataFrame({"probabilidad": [f"{prob[k] * 100:.0f} %" for k in prob],
                                   "detectado": ["sí" if probs[k] else "no" for k in prob]}, index=list(prob)), width="stretch")
        c2.markdown("**Reglas sobre el histograma (clásico)**")
        f = diag["features"]
        c2.dataframe(pd.DataFrame({"medida": [f"p99 {f['percentil_99']:.2f} (mín. {diagnosis.P99_MIN})",
                                              f"p1 {f['percentil_1']:.2f} (máx. {diagnosis.P1_MAX})",
                                              f"rango {f['percentil_99'] - f['percentil_1']:.2f}",
                                              f"bordes {f['dominante_bordes']:.2f} (umbral {diagnosis.COLOR_UMBRAL})"],
                                   "detectado": ["sí" if probs[k] else "no" for k in
                                                 ("oscura", "sobreexpuesta", "poco_contraste", "dominante_color")]},
                                  index=["oscura", "sobreexpuesta", "poco contraste", "dominante de color"]), width="stretch")
        plan = pipeline.plan(probs)
        st.subheader("2. Tratamiento decidido por el sistema")
        if not plan:
            st.success("El diagnóstico no encontró problemas claros: la foto se deja como está. "
                       "Si quieres otro ajuste, usa la pestaña Herramientas.")
        else:
            st.markdown(" → ".join(f"**{NOMBRES[p]}**" for p in plan))
            st.caption("Sin parámetros: el diagnóstico decide los pasos y cada paso mide la foto para decidir cuánto corregir. "
                       "Solo se usan nuestras redes y métodos clásicos (nada generativo).")
        if plan and st.button("Mejorar automáticamente", type="primary", key="auto_btn"):
            st.session_state["auto"] = img_key
        if plan and st.session_state.get("auto") == img_key:
            out, log = run_steps(work, tuple(plan))
            st.subheader("3. Resultado")
            before_after(work, out)
            st.dataframe(pd.DataFrame(log).set_index("paso").rename(index=NOMBRES), width="stretch")
            reference_metrics(reference, work, out)
            download(out, src_name, "dl_auto")

# ---------------- 2. Herramientas individuales ----------------
with tab_tools:
    st.caption("Activa las herramientas que quieras. Se aplican en el orden correcto: ruido → detalle → contraste → luz → color → nitidez → ampliar.")
    c1, c2, c3 = st.columns(3)
    t_noise = c1.selectbox("Reducir ruido", ["No", "Nuestra CNN", "Mediana", "Gaussiano"], key="t_ruido")
    t_detail = c1.selectbox("Restaurar detalle", ["No", "Nuestra CNN de detalle", "Real-ESRGAN (externo, generativo)"],
                            key="t_detalle")
    t_force = c1.slider("Intensidad de Real-ESRGAN", 0.0, 1.0, 0.7, 0.1, key="t_fuerza",
                        help="1 = modelo puro (a veces aspecto 'pintado'); menos = mezcla con la original.") \
        if t_detail.startswith("Real") else 0.7
    t_light = c2.selectbox("Iluminación", ["No", "Automática (gamma)", "Manual"], key="t_luz")
    gamma = c2.slider("γ (menor que 1 aclara)", 0.3, 3.0, 1.0, 0.05, key="t_gamma") if t_light == "Manual" else None
    t_contrast = c2.selectbox("Contraste", ["No", "Niveles automáticos", "CLAHE (por zonas)"], key="t_contraste")
    t_color = c3.slider("Balance de blancos (intensidad)", 0.0, 1.0, 0.0, 0.1, key="t_color")
    t_sharp = c3.slider("Nitidez (máscara de desenfoque)", 0.0, 2.0, 0.0, 0.1, key="t_nitidez")
    t_up = c3.selectbox("Ampliar resolución", ["No", "×2 — nuestra CNN", "×2 — Real-ESRGAN (externo)", "×4 — Real-ESRGAN (externo)"],
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
    before_after(work, out, cap_b="Resultado")
    if log:
        st.dataframe(pd.DataFrame(log).set_index("paso").rename(index=NOMBRES), width="stretch")
        reference_metrics(reference, work, out)
        download(out, src_name, "dl_tools")

# ---------------- 3. Análisis ----------------
with tab_an:
    c1, c2 = st.columns([3, 2])
    c1.image(rgb(work), caption="Imagen de entrada", width="stretch")
    c2.metric("Sigma de ruido ESTIMADO (sin referencia)", f"{analysis.estimate_noise_sigma(work):.1f}")
    c2.caption("Estimación de Immerkær (1996). Las texturas finas la inflan y el desenfoque la baja: no reemplaza al PSNR.")
    c2.dataframe(pd.DataFrame(analysis.channel_stats(work)).T.round(1), width="stretch")
    h = analysis.histograms(work)
    c2.line_chart(pd.DataFrame({"R": h["R"], "G": h["G"], "B": h["B"]}), color=["#e45756", "#54a24b", "#4c78a8"], height=220)
    with st.expander("Magnitud del gradiente (Sobel)"):
        g = analysis.gradient_magnitude(work)
        st.image(np.clip(g / (np.percentile(g, 99) + 1e-9), 0, 1), width="stretch")

# ---------------- 4. Comparación técnica (ruido) ----------------
with tab_tec:
    st.caption("Filtros clásicos contra nuestra CNN, con los parámetros elegidos en validación (results/classic_params.json).")
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
        c1, c2 = st.columns(2)
        c1.image(np.clip(0.5 + 3 * (work - results["Nuestra CNN"]), 0, 1)[..., ::-1],
                 caption="Lo que la CNN estimó como ruido (entrada − salida, ×3)", width="stretch")
        w = models["denoiser"].noise_net[0].weight.detach().numpy()[:, 1]
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)
        tiles = [cv2.resize(k, (48, 48), interpolation=cv2.INTER_NEAREST) for k in w]
        c2.image(np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]), clamp=True, width=380,
                 caption=f"32 filtros 3×3 aprendidos por la 1.ª capa (canal G) — época {models['meta'].get('epoch')}")
