import numpy as np
import pytest

torch = pytest.importorskip("torch")
from imageenhance import pipeline as P  # noqa: E402
from imageenhance.external import SRVGGNetCompact  # noqa: E402
from imageenhance.model import DenoiseCNN  # noqa: E402

SANA = {"ruido": False, "desenfoque": False, "compresion": False, "oscura": False,
        "sobreexpuesta": False, "poco_contraste": False, "dominante_color": False}


@pytest.fixture
def models():
    torch.manual_seed(0)
    return {"denoiser": DenoiseCNN(3, 8, 3), "esrgan": SRVGGNetCompact(feat=8, n_conv=2).eval()}


def test_plan_respeta_el_orden():
    assert P.plan(SANA) == []
    todo = dict.fromkeys(SANA, True)
    assert P.plan(todo) == ["ruido", "detalle", "luz", "contraste", "color"]
    assert P.plan({**SANA, "compresion": True, "oscura": True}) == ["detalle", "luz"]


def test_run_aplica_en_orden_y_registra_origen(img, models):
    out, log = P.run(img, ["color", "ruido", "luz"], models)
    assert out.shape == img.shape and 0 <= out.min() and out.max() <= 1
    assert [l["paso"] for l in log] == ["ruido", "luz", "color"]
    assert "Nuestra CNN" in log[0]["origen"]


def test_ampliar_y_respaldo_sin_modelo_externo(img, models):
    out, _ = P.run(img, ["ampliar"], models, factor=2)
    assert out.shape[:2] == (img.shape[0] * 2, img.shape[1] * 2)
    out, log = P.run(img, ["detalle"], {"denoiser": models["denoiser"], "esrgan": None})
    assert out.shape == img.shape and "no disponible" in log[0]["detalle"]
