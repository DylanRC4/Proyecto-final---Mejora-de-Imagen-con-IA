import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")
from streamlit.testing.v1 import AppTest  # noqa: E402

from imageenhance import data  # noqa: E402


def app():
    return AppTest.from_file(str(data.ROOT / "app.py"), default_timeout=180).run()


def test_app_arranca_sin_errores():
    assert not app().exception


def test_modo_automatico_sin_controles_y_foto_real_sin_psnr():
    at = app()
    if not at.radio:
        pytest.skip("Sin fotos de ejemplo")
    at.radio(key="modo").set_value("Experimento: degradar una foto limpia").run()  # ruido σ25: siempre hay plan
    assert not at.multiselect  # el usuario no elige pasos ni parámetros
    at.button(key="auto_btn").click().run()
    assert not at.exception and any("Nuestra CNN de ruido" in str(df.value) for df in at.dataframe)
    at.radio(key="modo").set_value("Foto real (sin referencia)").run()
    assert not any("PSNR (dB) ↑" in str(df.value.columns.tolist()) for df in at.dataframe)


def test_modo_experimento_muestra_psnr():
    at = app()
    at.radio(key="modo").set_value("Experimento: degradar una foto limpia").run()
    at.selectbox(key="t_ruido").set_value("Nuestra CNN").run()
    assert not at.exception
    assert any("PSNR (dB) ↑" in str(df.value.columns.tolist()) for df in at.dataframe)


def test_herramientas_con_nuestra_cnn_de_detalle_y_ampliar():
    at = app()
    at.selectbox(key="t_detalle").set_value("Nuestra CNN de detalle").run()
    at.selectbox(key="t_ampliar").set_value("×2 — nuestra CNN").run()
    assert not at.exception
    assert any("CNN de detalle" in str(df.value) for df in at.dataframe)
