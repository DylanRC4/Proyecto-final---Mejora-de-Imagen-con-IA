import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")
from streamlit.testing.v1 import AppTest  # noqa: E402

from imageenhance import data  # noqa: E402


def test_app_arranca_sin_errores():
    at = AppTest.from_file(str(data.ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception


def test_app_modo_foto_real_no_muestra_psnr():
    at = AppTest.from_file(str(data.ROOT / "app.py"), default_timeout=120).run()
    if not at.radio:
        pytest.skip("Sin fotos de ejemplo disponibles")
    at.radio[0].set_value("Foto real (sin referencia)").run()
    assert not at.exception
    assert not any("PSNR (dB)" in str(df.value.columns.tolist()) for df in at.dataframe)
