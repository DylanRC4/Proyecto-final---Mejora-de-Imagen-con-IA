import cv2
import numpy as np
import pytest

from imageenhance import classic, noise


@pytest.mark.parametrize("kernel", [classic.gaussian_kernel(5, 1.2), classic.box_kernel(3),
                                    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)])
def test_conv2d_propia_igual_a_opencv(img, kernel):
    ours = classic.conv2d(img, kernel)
    ref = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours - ref).max() < 1e-5


def test_conv2d_ejemplo_del_taller():
    # Sesión 04: media 3x3 sobre el 250 -> (1/9)*390 = 43.33
    a = np.array([[10, 20, 30], [40, 250, 0], [10, 20, 10]], np.float32)
    assert classic.conv2d(a, classic.box_kernel(3))[1, 1] == pytest.approx(390 / 9, rel=1e-5)


def test_kernel_gaussiano_igual_a_opencv():
    g = cv2.getGaussianKernel(7, 1.5)
    assert np.abs(classic.gaussian_kernel(7, 1.5) - g @ g.T).max() < 1e-6


def test_gaussiano_propio_igual_a_opencv(img):
    assert np.abs(classic.gaussian_filter(img, 5, 1.0, own=True) - classic.gaussian_filter(img, 5, 1.0)).max() < 1e-5


def test_mediana_elimina_sal_y_pimienta_mejor_que_gaussiano(img):
    sp = noise.add_salt_pepper(img, 0.05, seed=42)
    err = lambda x: np.abs(x - img).mean()
    assert err(classic.median_filter(sp, 3)) < err(classic.gaussian_filter(sp, 3, 0.8)) < err(sp)
    assert classic.median_filter(sp, 7).shape == img.shape
