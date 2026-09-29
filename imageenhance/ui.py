"""Presentación de Revela: estilos y piezas visuales de la interfaz. No procesa imágenes (eso es pipeline.py).

Colores de la Institución Universitaria de Colombia (azul #030568, amarillo #FACC15), definidos en
.streamlit/config.toml; aquí solo va lo que el tema de Streamlit no cubre (encabezado, chips, barras, pasos).
"""
import base64
from functools import cache
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st

AMARILLO = (250, 204, 21)  # RGB: línea del deslizador antes/después
LOGO = Path(__file__).resolve().parents[1] / "assets" / "logo_universitaria.png"
CSS = """<style>
[data-testid="stMainBlockContainer"] {padding-top: 2.2rem; max-width: 1440px}
header[data-testid="stHeader"] {background: transparent}
.rv-head {display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  border-bottom: 1px solid #D5D8E6; padding-bottom: .9rem; margin-bottom: .4rem}
.rv-brand {font-family: "Source Serif", Georgia, serif; font-weight: 700; font-size: 2.3rem;
  color: #030568; line-height: 1; letter-spacing: -.01em}
.rv-brand::after {content: ""; display: block; width: 2.4rem; height: .28rem; background: #FACC15; margin-top: .45rem}
.rv-tag {color: #4A4E75; font-size: .95rem; margin-top: .55rem}
.rv-meta {color: #4A4E75; font-size: .85rem; text-align: right}
.rv-logo {height: 3.3rem; display: block; margin: 0 0 .55rem auto}
.rv-meta b {color: #030568}
.rv-side-brand {font-family: "Source Serif", Georgia, serif; font-size: 1.9rem; font-weight: 700;
  color: #fff; line-height: 1}
.rv-side-brand::after {content: ""; display: block; width: 2rem; height: .25rem; background: #FACC15; margin: .4rem 0 .35rem}
.rv-side-sub {font-size: .78rem; color: #C9CBEA; line-height: 1.35}
.rv-label {font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #FACC15;
  margin: 1.1rem 0 .35rem}
.rv-section {font-size: .74rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #030568;
  margin: .2rem 0 .1rem}
.rv-section-sub {font-size: .82rem; color: #5A5E86; margin-bottom: .55rem}
.rv-model {display: flex; align-items: center; gap: .5rem; font-size: .84rem; color: #fff; padding: .18rem 0}
.rv-dot {width: .5rem; height: .5rem; border-radius: 50%; background: #FACC15; flex: none}
.rv-dot.off {background: #5B5FA8}
.rv-model small {margin-left: auto; color: #AEB1DD; font-size: .72rem}
.rv-team {font-size: .75rem; color: #AEB1DD; line-height: 1.5; margin-top: 1.6rem; border-top: 1px solid #2A2F9A;
  padding-top: .8rem}
.rv-bar {margin: .32rem 0}
.rv-bar-top {display: flex; justify-content: space-between; font-size: .85rem}
.rv-bar-top span:last-child {color: #5A5E86; font-variant-numeric: tabular-nums}
.rv-track {height: .38rem; background: #E3E5F0; border-radius: 1rem; overflow: hidden; margin-top: .15rem}
.rv-fill {height: 100%; background: #9CA0CC}
.rv-bar.on .rv-bar-top span:first-child {font-weight: 700; color: #030568}
.rv-bar.on .rv-fill {background: #FACC15}
.rv-flags {display: flex; flex-wrap: wrap; gap: .35rem; margin: .5rem 0 .2rem}
.rv-flag {font-size: .78rem; padding: .12rem .55rem; border-radius: 1rem; border: 1px solid #D5D8E6; color: #5A5E86}
.rv-flag.on {background: #FFF6D1; border-color: #FACC15; color: #15173D; font-weight: 600}
.rv-chips {display: flex; flex-wrap: wrap; align-items: center; gap: .35rem; margin: .3rem 0 .7rem}
.rv-chip {font-size: .82rem; padding: .22rem .65rem; border-radius: .25rem; font-weight: 600}
.rv-chip.propio {background: #030568; color: #fff}
.rv-chip.clasico {background: #fff; color: #030568; border: 1px solid #030568}
.rv-chip.externo {background: #FFF6D1; color: #15173D; border: 1px solid #FACC15}
.rv-arrow {color: #9CA0CC}
.rv-step {display: flex; gap: .8rem; padding: .6rem 0; border-bottom: 1px solid #E3E5F0}
.rv-num {width: 1.6rem; height: 1.6rem; border-radius: 50%; background: #030568; color: #FACC15; font-weight: 700;
  font-size: .8rem; display: flex; align-items: center; justify-content: center; flex: none}
.rv-step-body {flex: 1; font-size: .86rem; color: #3A3E66}
.rv-step-body b {color: #15173D; font-size: .92rem}
.rv-step-body small {color: #7A7EA6; float: right}
.rv-note {border-left: 3px solid #FACC15; background: #fff; padding: .55rem .8rem; font-size: .85rem; color: #3A3E66;
  margin: .5rem 0}
.rv-legend {display: flex; justify-content: space-between; font-size: .78rem; color: #5A5E86; margin-top: -.2rem}
[data-testid="stMetric"] {background: #fff; border: 1px solid #D5D8E6; border-left: 4px solid #FACC15;
  border-radius: .25rem; padding: .55rem .85rem}
[data-testid="stBaseButton-primary"]:hover {background: #1A1F8F; border-color: #1A1F8F}
[data-testid="stBaseButton-primary"]:focus:not(:active) {background: #030568; border-color: #FACC15; color: #fff}
[data-baseweb="tab-highlight"] {background-color: #FACC15; height: 3px}
[data-baseweb="tab"] p {font-weight: 600}
[data-testid="stImage"] img {border-radius: .2rem}
[data-testid="stFileUploaderDropzoneInstructions"] small {display: none}
</style>"""


def html(markup: str, where=st) -> None:
    where.markdown(markup, unsafe_allow_html=True)


def style() -> None:
    html(CSS)


@cache
def logo() -> str:
    """Logo de la universidad incrustado en base64 (Streamlit no sirve archivos sueltos sin configurarlo)."""
    return f'<img class="rv-logo" alt="Universitaria de Colombia" src="data:image/png;base64,' \
           f'{base64.b64encode(LOGO.read_bytes()).decode()}">' if LOGO.exists() else ""


def header(src_name: str, w: int, h: int, mode: str) -> None:
    html(f'<div class="rv-head"><div><div class="rv-brand">Revela</div><div class="rv-tag">Restauración de '
         f'fotografías con redes neuronales entrenadas por nosotros</div></div><div class="rv-meta">{logo()}<b>'
         f'{escape(src_name)}</b><br>{w} × {h} px · {escape(mode)}</div></div>')


def sidebar_brand() -> None:
    html('<div class="rv-side-brand">Revela</div><div class="rv-side-sub">Inteligencia Artificial II<br>'
         'Institución Universitaria de Colombia</div>', st.sidebar)


def label(text: str, where=st.sidebar) -> None:
    html(f'<div class="rv-label">{escape(text)}</div>', where)


def section(title: str, sub: str = "", where=st) -> None:
    html(f'<div class="rv-section">{escape(title)}</div>' + (f'<div class="rv-section-sub">{escape(sub)}</div>' if sub else ""),
         where)


def models_list(rows: list[tuple[str, bool, str]]) -> None:
    html("".join(f'<div class="rv-model"><span class="rv-dot{"" if ok else " off"}"></span>{escape(n)}'
                 f'<small>{escape(o)}</small></div>' for n, ok, o in rows), st.sidebar)


def team(names: list[str]) -> None:
    html('<div class="rv-team">Equipo<br>' + "<br>".join(escape(n) for n in names) + "</div>", st.sidebar)


def bars(probs: dict[str, float], detected: dict[str, bool], names: dict[str, str], where=st) -> None:
    """Probabilidades del MLP como barras; en amarillo las que pasan de 0,5 (problema detectado)."""
    html("".join(f'<div class="rv-bar{" on" if detected[k] else ""}"><div class="rv-bar-top"><span>{escape(names[k])}'
                 f'</span><span>{p * 100:.0f} %</span></div><div class="rv-track"><div class="rv-fill" '
                 f'style="width:{max(p * 100, 1):.0f}%"></div></div></div>' for k, p in probs.items()), where)


def flags(items: dict[str, bool], where=st) -> None:
    html('<div class="rv-flags">' + "".join(f'<span class="rv-flag{" on" if on else ""}">{escape(n)}</span>'
                                            for n, on in items.items()) + "</div>", where)


def kind(origin: str) -> str:
    """Clasifica un paso por su origen: nuestra IA, método clásico o modelo externo (Real-ESRGAN)."""
    return "externo" if "externo" in origin else "propio" if "Nuestra" in origin or "nuestra" in origin else "clasico"


def chips(steps: list[str], names: dict[str, str], origins: dict[str, str], where=st) -> None:
    arrow = '<span class="rv-arrow">→</span>'
    html('<div class="rv-chips">' + arrow.join(f'<span class="rv-chip {kind(origins.get(s, ""))}">{escape(names[s])}</span>'
                                               for s in steps) + "</div>", where)


def steps_log(log: list[dict], names: dict[str, str], where=st) -> None:
    """Qué hizo el sistema, paso a paso: nombre, origen, detalle medido y tiempo."""
    def row(i, r):
        secs = f"<small>{r['segundos']:.2f} s</small>" if "segundos" in r else ""
        extra = "<br>" + escape(str(r["detalle"])) if r.get("detalle") else ""
        return (f'<div class="rv-step"><div class="rv-num">{i}</div><div class="rv-step-body"><b>'
                f'{escape(names.get(r["paso"], r["paso"]))}</b>{secs}<br>{escape(r["origen"])}{extra}</div></div>')
    html("".join(row(i, r) for i, r in enumerate(log, 1)), where)


def note(text: str, where=st) -> None:
    html(f'<div class="rv-note">{escape(text)}</div>', where)


def split(a: np.ndarray, b: np.ndarray, pos: int) -> np.ndarray:
    """Deslizador antes/después: a la izquierda del corte la original, a la derecha el resultado (RGB uint8)."""
    x, lw = a.shape[1] * pos // 100, max(2, a.shape[1] // 320)
    out = b.copy()
    out[:, :x] = a[:, :x]
    out[:, max(x - lw // 2, 0):x + lw - lw // 2] = AMARILLO
    return out
