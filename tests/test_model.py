import numpy as np
import pytest

torch = pytest.importorskip("torch")
from imageenhance import classic, model as M  # noqa: E402


def test_parametros_coinciden_con_la_formula():
    net = M.DenoiseCNN(3, 32, 7)
    assert M.count_params(net) == M.formula_params(3, 32, 7) == 48003


def test_conv2d_numpy_es_la_misma_operacion_que_nn_conv2d(img):
    """El filtro fijo que programamos y la capa de PyTorch hacen la misma cuenta; lo que cambia es quién elige los pesos."""
    k = classic.gaussian_kernel(3, 1.0)
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect", bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.from_numpy(k)[None, None]
        out = conv(torch.from_numpy(img[..., 0])[None, None])[0, 0].numpy()
    assert np.abs(out - classic.conv2d(img[..., 0], k)).max() < 1e-5


def test_salida_misma_forma_y_residual_con_ruido_cero(img):
    net = M.DenoiseCNN(3, 8, 3)
    with torch.no_grad():
        for p in net.noise_net[-1].parameters():
            p.zero_()  # si la red estima ruido 0, la salida debe ser la entrada
    assert np.allclose(M.denoise(net, img), img, atol=1e-6)


def test_procesar_por_bloques_igual_que_completa(img):
    torch.manual_seed(0)
    net = M.DenoiseCNN(3, 8, 5)
    full = M.denoise(net, img, tile=10_000)
    assert np.abs(M.denoise(net, img, tile=24, pad=8) - full).max() < 1e-5


def test_guardar_y_cargar(tmp_path, img):
    net = M.DenoiseCNN(3, 8, 3)
    M.save(net, tmp_path / "m.pt", epoch=1, val_psnr=30.5)
    net2, meta = M.load(tmp_path / "m.pt")
    assert meta["epoch"] == 1 and np.allclose(M.denoise(net, img), M.denoise(net2, img))
