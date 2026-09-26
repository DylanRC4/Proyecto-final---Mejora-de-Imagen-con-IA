import numpy as np
import pytest

from imageenhance import diagnosis as D
from imageenhance import noise


@pytest.fixture
def foto():
    """Imagen sintética más grande con bordes y texturas, 256x320."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:256, 0:320].astype(np.float32)
    base = 0.3 + 0.3 * (np.sin(x / 7) * np.cos(y / 11) > 0) + 0.1 * np.sin(x / 2.5)
    return np.clip(np.stack([base, base * 0.9 + 0.05, base * 1.1], -1) + rng.normal(0, 0.005, (256, 320, 3)), 0, 1).astype(np.float32)


def test_vector_de_caracteristicas(foto):
    f = D.features(foto)
    assert f.shape == (len(D.FEATURES),) and np.isfinite(f).all()


def test_cada_degradacion_mueve_su_caracteristica(foto):
    f0 = D.features(foto)
    assert D.features(noise.add_gaussian(foto, 25, seed=0))[0] > f0[0] + 10  # más ruido estimado
    assert D.features(noise.add_blur(foto, 2.0))[1] < f0[1] - 0.5           # menos nitidez
    assert D.features(noise.add_jpeg(foto, 10))[4] > f0[4] + 0.3             # más bloques 8x8


def test_reglas_de_tono(foto):
    assert D.tone_flags(D.features(foto * 0.3))["oscura"]
    assert D.tone_flags(D.features(0.45 + 0.1 * foto))["poco_contraste"]
    assert D.tone_flags(D.features(np.clip(foto * np.array([0.6, 1.0, 1.5], np.float32), 0, 1)))["dominante_color"]
    assert not D.tone_flags(D.features(foto))["oscura"]


def test_mlp_tiene_963_parametros():
    torch = pytest.importorskip("torch")
    assert sum(p.numel() for p in D.build_mlp().parameters()) == 963
