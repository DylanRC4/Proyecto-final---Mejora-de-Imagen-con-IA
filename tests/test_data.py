import numpy as np
import pytest

from imageenhance import data


def test_semilla_estable_y_distinta_por_condicion():
    a = data.noise_seed("12003.jpg", "gaussian", 25)
    assert a == data.noise_seed("12003.jpg", "gaussian", 25)
    assert a != data.noise_seed("12003.jpg", "gaussian", 50) != data.noise_seed("12074.jpg", "gaussian", 50)


def test_split_sin_fotos_compartidas():
    s = {k: set(data.split_names(k)) for k in data.SPLITS}
    assert len(s["train"]) == 200 and len(s["val"]) == 100 and len(s["test"]) == 200
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"])


def test_parches_forma_y_reproducibles(img):
    a = data.random_patches(img, 5, 32, np.random.default_rng(0))
    b = data.random_patches(img, 5, 32, np.random.default_rng(0))
    assert a.shape == (5, 32, 32, 3) and np.array_equal(a, b)
