# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from coltap import synthetic


@pytest.fixture(scope="session")
def small_scene():
    textures = [synthetic.procedural_texture(256, seed=s) for s in range(4)]
    return synthetic.Scene(textures, texels_per_unit=64.0)


@pytest.fixture(scope="session")
def small_camera():
    return synthetic.Camera(width=160, height=120, focal=130.0)


def random_points_in_view(rng, num, depth=(3.0, 6.0), spread=1.5):
    return np.stack(
        [
            rng.uniform(-spread, spread, num),
            rng.uniform(-spread, spread, num),
            rng.uniform(*depth, num),
        ],
        axis=1,
    )
