import numpy as np

from imageenhance import analysis, noise


def test_estimacion_de_ruido_sigue_al_sigma_real():
    y, x = np.mgrid[0:200, 0:200].astype(np.float32)
    smooth = np.repeat((0.3 + 0.4 * x / 199)[..., None], 3, axis=-1)  # rampa: el kernel la anula
    assert analysis.estimate_noise_sigma(smooth) < 0.5
    for s in (10, 25):
        est = analysis.estimate_noise_sigma(noise.add_gaussian(smooth, s, seed=0, clip=False))
        assert abs(est - s) < 0.1 * s


def test_histogramas_y_estadisticas(img):
    h = analysis.histograms(img)
    assert set(h) == {"B", "G", "R"} and h["G"].sum() == img.shape[0] * img.shape[1]
    assert 0 <= analysis.channel_stats(img)["R"]["media"] <= 255


def test_sobel_borde_vertical():
    a = np.zeros((5, 6, 3), np.float32)
    a[:, 3:] = 1.0
    g = analysis.gradient_magnitude(a)
    assert g[2, 2] > 0 and g[2, 0] == 0
