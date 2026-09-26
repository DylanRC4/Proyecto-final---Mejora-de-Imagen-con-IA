import numpy as np
import pytest

torch = pytest.importorskip("torch")
from imageenhance import external as E  # noqa: E402


def test_arquitectura_tiene_los_parametros_del_modelo_publicado():
    net = E.SRVGGNetCompact()
    first, mid, last = 3 * 64 * 9 + 64 + 64, 32 * (64 * 64 * 9 + 64 + 64), 64 * 48 * 9 + 48
    assert sum(p.numel() for p in net.parameters()) == first + mid + last == 1_213_296


def test_amplia_x4_y_por_bloques_igual_que_completa(img):
    torch.manual_seed(0)
    net = E.SRVGGNetCompact(feat=8, n_conv=2).eval()
    full = E.upscale4(net, img, tile=10_000)
    assert full.shape == (img.shape[0] * 4, img.shape[1] * 4, 3)
    assert np.abs(E.upscale4(net, img, tile=24, pad=8) - full).max() < 1e-5
    assert E.restore(net, img).shape == img.shape and E.upscale(net, img, 2).shape[0] == img.shape[0] * 2


@pytest.mark.skipif(not E.ESRGAN_PATH.exists(), reason="modelo externo no descargado")
def test_pesos_reales_cargan():
    net = E.load_esrgan()
    assert sum(p.numel() for p in net.parameters()) == 1_213_296
