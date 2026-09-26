import numpy as np
import pytest


@pytest.fixture
def img():
    """Imagen sintética 64x80x3 en [0, 1]: degradado suave + un borde vertical + textura."""
    y, x = np.mgrid[0:64, 0:80].astype(np.float32)
    base = 0.2 + 0.5 * (x / 79)
    base[:, 40:] += 0.2
    tex = 0.05 * np.sin(x / 3) * np.cos(y / 4)
    return np.clip(np.stack([base + tex, base, base - tex], axis=-1), 0, 1).astype(np.float32)
