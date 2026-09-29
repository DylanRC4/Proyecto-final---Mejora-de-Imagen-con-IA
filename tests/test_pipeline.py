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
    return {"denoiser": DenoiseCNN(3, 8, 3), "detail": DenoiseCNN(3, 8, 3), "esrgan": SRVGGNetCompact(feat=8, n_conv=2).eval()}


def test_retoque_reemplaza_luz_y_contraste_y_arranca_como_identidad(models):
    from imageenhance.retouch import RetouchNet
    oscura = {**SANA, "ruido": True, "oscura": True}
    assert P.plan(oscura) == ["ruido", "contraste", "luz"] and P.plan(oscura, retoque=True) == ["ruido", "retoque"]
    img = np.random.default_rng(0).random((40, 50, 3)).astype(np.float32)
    out, log = P.run(img, ["retoque"], {**models, "retouch": RetouchNet()})
    assert np.abs(out - img).max() < 1e-5 and "FiveK" in log[0]["origen"]


def test_plan_respeta_el_orden():
    assert P.plan(SANA) == []
    todo = dict.fromkeys(SANA, True)
    assert P.plan(todo) == ["ruido", "detalle", "contraste", "luz"]  # el color solo se avisa, no se corrige solo
    assert P.plan({**SANA, "compresion": True, "oscura": True}) == ["detalle", "contraste", "luz"]


def test_run_aplica_en_orden_y_registra_origen(img, models):
    out, log = P.run(img, ["color", "ruido", "luz"], models)
    assert out.shape == img.shape and 0 <= out.min() and out.max() <= 1
    assert [l["paso"] for l in log] == ["ruido", "luz", "color"]  # luz y color se omiten si la foto ya está bien
    assert "Nuestra CNN" in log[0]["origen"]


def test_modo_automatico_solo_usa_lo_nuestro(img, models):
    assert not any(p.endswith("_externo") for p in P.plan(dict.fromkeys(SANA, True)))
    out, log = P.run(img, ["ruido", "detalle"], {**models, "esrgan": None})  # funciona sin el modelo externo
    assert out.shape == img.shape and all("Nuestra CNN" in l["origen"] for l in log)


def test_ampliar_nuestra_x2_y_externa_x4(img, models):
    out, log = P.run(img, ["ampliar"], models, factor=4)  # la nuestra siempre es ×2
    assert out.shape[:2] == (img.shape[0] * 2, img.shape[1] * 2) and "nuestra CNN" in log[0]["origen"]
    out, _ = P.run(img, ["ampliar_externo"], models, factor=4)
    assert out.shape[:2] == (img.shape[0] * 4, img.shape[1] * 4)
    with pytest.raises(RuntimeError, match="download_models"):
        P.run(img, ["ampliar_externo"], {**models, "esrgan": None})


def test_respaldo_sin_cnn_de_detalle_e_intensidad_cero_externa(img, models):
    out, log = P.run(img, ["detalle"], {**models, "detail": None})
    assert out.shape == img.shape and "falta" in log[0]["detalle"]
    out, _ = P.run(img, ["detalle_externo"], models, strength=0.0)
    assert np.allclose(out, img)


def test_tono_vuelve_a_medir_y_no_deja_velo_gris(img):
    oscura = 0.3 * img + 0.05  # subexpuesta y con negros levantados
    out, log = P.run(oscura, ["contraste", "luz"], {})
    assert log[0]["paso"] == "contraste" and "no hizo falta" in log[1]["detalle"]  # los niveles ya bastaron
    assert np.percentile(out, 1) < 0.05  # los negros vuelven a ser negros
    bien, log = P.run(img, ["luz", "color"], {})
    assert np.array_equal(bien, img) and all("no hizo falta" in l["detalle"] for l in log)
