import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from imageenhance import classic, metrics, noise


def test_psnr_casos_conocidos(img):
    assert metrics.psnr(img, img) == float("inf")
    # MSE = 0.01 -> PSNR = 20 dB exactos
    assert metrics.psnr(np.zeros(4), np.full(4, 0.1)) == pytest.approx(20.0)


@pytest.mark.parametrize("sigma", [10, 25, 50])
def test_psnr_y_ssim_iguales_a_scikit_image(img, sigma):
    noisy = noise.add_gaussian(img, sigma, seed=1)
    assert metrics.psnr(img, noisy) == pytest.approx(peak_signal_noise_ratio(img, noisy, data_range=1.0), abs=1e-4)
    ref = structural_similarity(img, noisy, data_range=1.0, channel_axis=-1, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False)
    assert metrics.ssim(img, noisy) == pytest.approx(ref, abs=1e-4)


def test_ssim_identica_y_ordena_calidad(img):
    assert metrics.ssim(img, img) == pytest.approx(1.0)
    n = noise.add_gaussian(img, 25, seed=3)
    assert metrics.ssim(img, classic.gaussian_filter(n, 5, 1.0)) > metrics.ssim(img, n)
