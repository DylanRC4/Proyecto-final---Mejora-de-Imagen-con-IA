import cv2
import numpy as np

from imageenhance import io_utils as io
from imageenhance import noise


def test_uint8_float_ida_y_vuelta(img):
    u8 = io.to_uint8(img)
    assert u8.dtype == np.uint8
    assert np.abs(io.to_float(u8) - img).max() <= 0.5 / 255 + 1e-6


def test_to_uint8_hace_clip_y_no_da_la_vuelta():
    assert io.to_uint8(np.array([-0.04, 1.2], np.float32)).tolist() == [0, 255]


def test_bgr_rgb_invierte_canales(img):
    rgb = io.bgr_to_rgb(img)
    assert np.array_equal(rgb[..., 0], img[..., 2]) and np.array_equal(io.bgr_to_rgb(rgb), img)


def test_gris_coincide_con_opencv(img):
    u8 = io.to_uint8(img)
    diff = np.abs(io.to_gray(u8) - cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float32))
    assert diff.max() <= 0.5 + 1e-3  # solo redondeo, como en la Sesión 02


def test_guardar_y_cargar_con_tildes(tmp_path, img):
    p = tmp_path / "prueba_señal.png"
    p.write_bytes(cv2.imencode(".png", io.to_uint8(img))[1].tobytes())
    assert np.array_equal(io.load_bgr(p), io.to_uint8(img))


def test_gaussiano_reproducible_y_con_sigma_correcto(img):
    a = noise.add_gaussian(img, 25, seed=7, clip=False)
    b = noise.add_gaussian(img, 25, seed=7, clip=False)
    c = noise.add_gaussian(img, 25, seed=8, clip=False)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs((a - img).std() * 255 - 25) < 1.0
    clipped = noise.add_gaussian(img, 50, seed=7)
    assert clipped.min() >= 0 and clipped.max() <= 1


def test_sal_pimienta_proporcion_y_no_modifica_original(img):
    orig = img.copy()
    sp = noise.add_salt_pepper(img, 0.05, seed=42)
    frac = np.any(sp != img, axis=-1).mean()
    assert np.array_equal(img, orig)
    assert 0.03 < frac < 0.06


def test_degradaciones_de_detalle(img):
    assert noise.add_downscale(img, 2.0).shape == img.shape
    assert noise.fixed_detail(img).shape == img.shape
    a, b = noise.random_detail(img, np.random.default_rng(5)), noise.random_detail(img, np.random.default_rng(5))
    assert np.array_equal(a, b)  # misma semilla, misma degradación
    # reducir y ampliar pierde detalle fino: la imagen cambia, pero su brillo medio casi no
    d = noise.add_downscale(img, 3.0)
    assert np.abs(d - img).mean() > 1e-3 and abs(d.mean() - img.mean()) < 0.01


def test_mezcla_de_detalle_v2_equilibrada(img):
    rng = np.random.default_rng(0)
    iguales = sum(np.array_equal(noise.random_detail_v2(img, rng), img) for _ in range(200))
    assert 10 <= iguales <= 35  # ~10 % de fotos limpias: la red aprende a no tocarlas
    assert all(f(img).shape == img.shape for f in noise.VAL_DETAIL)


def test_exposicion_fisica_y_dominante(img):
    assert np.allclose(noise.add_exposure(img, 0), img, atol=1e-5)
    osc, quem = noise.add_exposure(img, -2), noise.add_exposure(img, 2)
    assert osc.mean() < 0.6 * img.mean() and quem.max() > 0.999 and quem.mean() > img.mean()
    cast = noise.add_cast(img).reshape(-1, 3).mean(0)
    assert cast[0] < img.reshape(-1, 3).mean(0)[0] and cast[2] > img.reshape(-1, 3).mean(0)[2] * 1.05


def test_ruido_de_camara_reproducible_y_mas_fuerte_en_sombras(img):
    a = noise.add_camera_noise(img, np.random.default_rng(0), 3e-3)
    assert a.shape == img.shape and np.array_equal(a, noise.add_camera_noise(img, np.random.default_rng(0), 3e-3))
    plano = lambda v: noise.add_camera_noise(np.full((64, 64, 3), v, np.float32), np.random.default_rng(1), 3e-3).std()
    assert plano(0.1) > plano(0.8)  # en sRGB las sombras quedan más ruidosas, como en una foto nocturna


def test_mezcla_v3_y_ruido_v2(img):
    rng = np.random.default_rng(0)
    assert 5 <= sum(np.array_equal(noise.random_detail_v3(img, rng), img) for _ in range(200)) <= 30
    assert all(f(img, 3).shape == img.shape for f in noise.VAL_DETAIL_V3 + noise.VAL_NOISE_V2)
    assert noise.random_noise_v2(img, rng).shape == img.shape
