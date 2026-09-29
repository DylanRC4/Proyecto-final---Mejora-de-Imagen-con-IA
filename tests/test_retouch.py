import numpy as np
import pytest

torch = pytest.importorskip("torch")
from imageenhance import retouch as R  # noqa: E402


def test_red_nueva_no_cambia_la_foto_y_tiene_65517_parametros():
    net = R.RetouchNet()
    x = torch.rand(2, 3, 50, 70)
    assert sum(p.numel() for p in net.parameters()) == 65517
    assert (net(x) - x).abs().max() < 1e-5


def test_ajustes_globales_monotonos_y_en_rango():
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 256).repeat(3, 1)[None, :, None, :]  # rampa de gris
    assert 0 <= R.adjust(x, torch.randn(1, R.N_PARAMS) * 2).min() and R.adjust(x, torch.randn(1, R.N_PARAMS) * 2).max() <= 1
    p = torch.zeros(1, R.N_PARAMS)
    p[0, 4:4 + R.KNOTS] = torch.randn(R.KNOTS) * 2  # solo la curva de tono
    y = R.adjust(x, p)[0, 1, 0]
    assert (y[1:] - y[:-1]).min() >= -1e-6  # la curva nunca invierte el orden de los tonos


def test_retoque_de_foto_de_cualquier_tamano():
    img = np.random.default_rng(0).random((123, 77, 3)).astype(np.float32)
    out, texto = R.retouch(R.RetouchNet(), img)
    assert out.shape == img.shape and "EV" in texto
