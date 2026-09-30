"""Shared fixtures: small analytic shapes with known metric values."""

import numpy as np
import pytest


def cube(shape=(40, 40, 40), lo=(10, 10, 10), size=(16, 16, 16)):
    m = np.zeros(shape, bool)
    m[lo[0]:lo[0] + size[0], lo[1]:lo[1] + size[1], lo[2]:lo[2] + size[2]] = True
    return m


def sphere(shape=(48, 48, 48), center=None, radius=10.0, spacing=(1, 1, 1)):
    center = np.array(center if center is not None else [s / 2 for s in shape])
    g = np.indices(shape).astype(float)
    d2 = sum(((g[i] - center[i]) * spacing[i]) ** 2 for i in range(3))
    return d2 <= radius ** 2


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    g = sphere(radius=11)
    p = np.roll(sphere(radius=10), 2, axis=0)
    p[rng.random(p.shape) < 0.0005] = True  # sparse speckle
    return p, g
