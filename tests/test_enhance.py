import numpy as np

from imageenhance import enhance
from imageenhance.io_utils import to_gray


def test_gamma_aclara_foto_oscura_y_oscurece_foto_quemada(img):
    oscura, g1 = enhance.auto_gamma(img * 0.25)
    quemada, g2 = enhance.auto_gamma(np.clip(img * 0.3 + 0.7, 0, 1))
    assert g1 < 1 < g2
    assert abs(to_gray(oscura).mean() - 0.45) < 0.08
    assert to_gray(quemada).mean() < to_gray(np.clip(img * 0.3 + 0.7, 0, 1)).mean() - 0.1  # γ se limita a 2.5


def test_auto_levels_estira_el_histograma(img):
    apagada = 0.45 + 0.1 * img  # contraste muy bajo: valores entre 0.45 y 0.55
    out, alpha, beta = enhance.auto_levels(apagada)
    g = to_gray(out)
    assert alpha > 5 and g.min() < 0.02 and g.max() > 0.98


def test_clahe_aumenta_contraste_y_conserva_forma():
    textura = np.random.default_rng(0).random((256, 320, 1), dtype=np.float32).repeat(3, axis=-1)
    apagada = 0.45 + 0.1 * textura  # textura con contraste muy bajo
    out = enhance.clahe(apagada)
    assert out.shape == apagada.shape and out.dtype == np.float32
    assert to_gray(out).std() > 1.2 * to_gray(apagada).std()


def test_balance_de_blancos_quita_dominante_de_color(img):
    amarilla = np.clip(img * np.array([0.8, 1.0, 1.2], np.float32), 0, 1)  # BGR: menos azul, más rojo
    out, gains = enhance.white_balance(amarilla)
    m = out.reshape(-1, 3).mean(axis=0)
    assert gains[0] > 1 > gains[2] and m.max() - m.min() < 0.3 * (np.ptp(amarilla.reshape(-1, 3).mean(0)))


def test_nitidez_no_cambia_zonas_planas_y_aumenta_bordes():
    flat = np.full((20, 20, 3), 0.5, np.float32)
    assert np.allclose(enhance.unsharp(flat), flat)
    borde = np.zeros((20, 20, 3), np.float32)
    borde[:, 10:] = 0.6
    out = enhance.unsharp(borde + 0.2)
    assert out[5, 10, 0] > 0.8 and out[5, 9, 0] < 0.2  # más contraste justo en el borde
