"""ImageEnhance AI — interfaz de demostración.   Ejecutar:  streamlit run app.py

Convención: todo se procesa en float32 [0, 1] y en orden BGR (como lo carga OpenCV y como se
entrenó la CNN); solo se convierte a RGB para mostrar (Sesión 02).
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from imageenhance import analysis, classic, metrics, noise
from imageenhance.io_utils import bgr_to_rgb, decode_bytes, load_bgr, resize_max, to_float, to_uint8

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "denoise_cnn.pt"
EXAMPLES = ROOT / "data" / "raw" / "bsds500" / "test"

st.set_page_config(page_title="ImageEnhance AI", layout="wide")


@st.cache_resource(show_spinner="Cargando la CNN…")
def load_model(path: str, mtime: float):
    from imageenhance import model as M
    return M.load(path)


@st.cache_data(show_spinner="Procesando con la CNN…")
def run_cnn(img: np.ndarray, mtime: float) -> tuple[np.ndarray, float]:
    from imageenhance import model as M
    net, _ = load_model(str(MODEL_PATH), mtime)
    t0 = time.perf_counter()
    out = M.denoise(net, img)
    return out, time.perf_counter() - t0


@st.cache_data
def run_classic(img: np.ndarray, method: str, params: tuple, own: bool = False) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    kw = dict(params)
    out = classic.gaussian_filter(img, own=own, **kw) if method == "gaussian" else classic.apply(img, method, **kw)
    return out, time.perf_counter() - t0


def rgb(img: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(to_uint8(img))


def classic_defaults() -> dict:
    p = ROOT / "results" / "classic_params.json"
    return json.loads(p.read_text("utf-8"))["best"] if p.exists() else {}


# ---------------- Barra lateral: entrada y modo ----------------
st.sidebar.title("ImageEnhance AI")
up = st.sidebar.file_uploader("Cargar fotografía", type=["jpg", "jpeg", "png", "bmp", "webp"])
examples = sorted(p.name for p in EXAMPLES.glob("*.jpg"))[:30] if EXAMPLES.exists() else []
if up is not None:
    src, src_name = decode_bytes(up.getvalue()), up.name
elif examples:
    src_name = st.sidebar.selectbox("…o usar una foto de prueba (BSDS500)", examples)
    src = load_bgr(EXAMPLES / src_name)
else:
    st.info("Carga una fotografía en la barra lateral para comenzar.")
    st.stop()

max_side = st.sidebar.slider("Lado máximo para procesar (px)", 256, 1600, 800, 64,
                             help="La CNN corre en CPU: imágenes más pequeñas se procesan más rápido.")
mode = st.sidebar.radio("Modo", ["Experimento: agregar ruido controlado", "Foto real (sin referencia)"])
experiment = mode.startswith("Experimento")
if experiment:
    kind = st.sidebar.selectbox("Tipo de ruido", ["gaussian", "salt_pepper"],
                                format_func=lambda k: {"gaussian": "Gaussiano", "salt_pepper": "Sal y pimienta"}[k])
    level = (st.sidebar.slider("Sigma (escala 0-255)", 5, 50, 25) if kind == "gaussian"
             else st.sidebar.slider("Fracción de píxeles", 0.01, 0.20, 0.05, 0.01))
    seed = int(st.sidebar.number_input("Semilla", 0, 10_000, 42))

original = to_float(resize_max(src, max_side))
if experiment:
    reference, work = original, noise.degrade(original, kind, level, seed)
else:
    reference, work = None, original

st.title("ImageEnhance AI")
st.caption("Eliminación de ruido con filtros clásicos y una CNN residual entrenada desde cero. "
           f"Imagen: {src_name} — {work.shape[1]}×{work.shape[0]} px.")
if experiment and kind == "salt_pepper":
    st.warning("La CNN se entrenó solo con ruido gaussiano: con sal y pimienta se espera que la mediana gane.")

tab_a, tab_c, tab_n, tab_cmp, tab_adj = st.tabs(["1. Análisis", "2. Filtros clásicos", "3. CNN", "4. Comparación", "5. Ajustes y descarga"])

# ---------------- 1. Análisis ----------------
with tab_a:
    c1, c2 = st.columns([3, 2])
    c1.image(rgb(work), caption="Imagen de entrada" + (" (con ruido agregado)" if experiment else ""), width="stretch")
    est = analysis.estimate_noise_sigma(work)
    c2.metric("Sigma de ruido ESTIMADO (sin referencia)", f"{est:.1f}",
              delta=f"real: {level}" if experiment and kind == "gaussian" else None, delta_color="off")
    c2.caption("Estimación de Immerkær (1996): convolución con un kernel que anula zonas planas. "
               "Las texturas finas la inflan; no reemplaza al PSNR.")
    c2.dataframe(pd.DataFrame(analysis.channel_stats(work)).T.round(1), width="stretch")
    h = analysis.histograms(work)
    c2.line_chart(pd.DataFrame({"R": h["R"], "G": h["G"], "B": h["B"]}), color=["#e45756", "#54a24b", "#4c78a8"], height=220)
    with st.expander("Magnitud del gradiente (Sobel, Sesión 05)"):
        g = analysis.gradient_magnitude(work)
        st.image(np.clip(g / (np.percentile(g, 99) + 1e-9), 0, 1), caption="El ruido también produce gradientes: "
                 "por eso Canny suaviza antes de derivar.", width="stretch")

# ---------------- 2. Filtros clásicos ----------------
defaults = classic_defaults()
key = f"{kind}_{level}" if experiment else "gaussian_25"
d_med, d_gau = defaults.get(f"{key}_median", {"ksize": 3}), defaults.get(f"{key}_gaussian", {"ksize": 7, "sigma": 0.8})
with tab_c:
    c1, c2 = st.columns(2)
    k_med = c1.select_slider("Mediana: tamaño del kernel", [3, 5, 7, 9], d_med["ksize"])
    k_gau = c2.select_slider("Gaussiano: tamaño del kernel", [3, 5, 7, 9, 11, 13, 15], d_gau["ksize"])
    s_gau = c2.slider("Gaussiano: sigma", 0.3, 3.0, float(d_gau["sigma"]), 0.1)
    own = c2.checkbox("Usar nuestra conv2d en NumPy (Sesión 01)")
    med, t_med = run_classic(work, "median", (("ksize", k_med),))
    gau, t_gau = run_classic(work, "gaussian", (("ksize", k_gau), ("sigma", s_gau)), own)
    c1.image(rgb(med), caption=f"Mediana {k_med}×{k_med} — {t_med * 1000:.0f} ms", width="stretch")
    c2.image(rgb(gau), caption=f"Gaussiano {k_gau}×{k_gau}, σ={s_gau} — {t_gau * 1000:.0f} ms "
             f"({'NumPy propio' if own else 'OpenCV'})", width="stretch")
    st.caption("Si hay un resultado de ajuste en validación (results/classic_params.json) se usan esos parámetros por defecto.")
    with st.expander("Kernel gaussiano que se está aplicando (filtro FIJO, diseñado a mano)"):
        st.dataframe(pd.DataFrame(classic.gaussian_kernel(k_gau, s_gau)).round(4))

# ---------------- 3. CNN ----------------
cnn = None
with tab_n:
    if not MODEL_PATH.exists():
        st.warning("Aún no hay modelo entrenado. Ejecuta: python scripts/train.py")
    else:
        mtime = MODEL_PATH.stat().st_mtime
        cnn, t_cnn = run_cnn(work, mtime)
        net, meta = load_model(str(MODEL_PATH), mtime)
        from imageenhance.model import count_params
        c1, c2 = st.columns(2)
        c1.image(rgb(cnn), caption=f"CNN — {t_cnn:.2f} s en CPU", width="stretch")
        resid = work - cnn
        c2.image(np.clip(0.5 + 3 * resid, 0, 1)[..., ::-1], caption="Lo que la red estimó como ruido "
                 "(entrada − salida, amplificado ×3)", width="stretch")
        st.caption(f"Parámetros: {count_params(net):,} | época guardada: {meta.get('epoch')} | "
                   f"PSNR de validación (σ={meta.get('config', {}).get('val_sigma')}): {meta.get('val_psnr', 0):.2f} dB")
        with st.expander("Filtros APRENDIDOS de la primera capa (32 kernels 3×3, canal G)"):
            w = net.noise_net[0].weight.detach().numpy()[:, 1]
            w = (w - w.min()) / (w.max() - w.min() + 1e-9)
            tiles = [cv2.resize(k, (48, 48), interpolation=cv2.INTER_NEAREST) for k in w]
            grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
            st.image(grid, caption="Nadie los diseñó: salen del descenso de gradiente.", width=400, clamp=True)

# ---------------- 4. Comparación ----------------
results = {"Entrada": work, "Mediana": med, "Gaussiano": gau}
if cnn is not None:
    results["CNN"] = cnn
with tab_cmp:
    cols = st.columns(len(results))
    for col, (name, im) in zip(cols, results.items()):
        col.image(rgb(im), caption=name, width="stretch")
    if reference is not None:
        rows = {n: metrics.evaluate(reference, im) for n, im in results.items()}
        st.subheader("Métricas con referencia limpia")
        st.dataframe(pd.DataFrame(rows).T.rename(columns={"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑"}).round(4),
                     width="stretch")
        st.caption("Válidas porque el ruido lo agregamos nosotros y conocemos la imagen original.")
    else:
        st.subheader("Sin referencia limpia")
        st.info("En una foto real no existe la imagen limpia, así que PSNR y SSIM NO se pueden calcular. "
                "Solo se muestra el sigma de ruido estimado que queda en cada resultado (estimación, no métrica).")
        st.dataframe(pd.DataFrame({n: {"sigma estimado": analysis.estimate_noise_sigma(im)} for n, im in results.items()}).T.round(2))

# ---------------- 5. Ajustes tradicionales y descarga ----------------
with tab_adj:
    base_name = st.selectbox("Resultado base", list(results)[::-1])
    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("Contraste (α)", 0.5, 2.0, 1.0, 0.05)
    beta = c2.slider("Brillo (β, escala 0-255)", -100, 100, 0, 5)
    amount = c3.slider("Nitidez (realce laplaciano)", 0.0, 2.0, 0.0, 0.1)
    final = classic.adjust_brightness_contrast(results[base_name], alpha, beta)
    if amount > 0:
        final = classic.sharpen(final, amount)
    st.image(rgb(final), caption=f"{base_name} + A' = {alpha}·A + {beta}" + (f" + realce {amount}" if amount else ""),
             width="stretch")
    st.caption("Estos ajustes son procesamiento TRADICIONAL (transformación afín y kernel de realce, Sesión 01), "
               "no los hace la CNN. El realce amplifica el ruido: aplícalo después de eliminarlo.")
    ok, buf = cv2.imencode(".png", to_uint8(final))
    st.download_button("Descargar PNG", buf.tobytes(), file_name=f"{Path(src_name).stem}_{base_name.lower()}.png",
                       mime="image/png")
