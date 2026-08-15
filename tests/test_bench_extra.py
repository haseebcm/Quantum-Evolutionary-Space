import numpy as np

from qes.bench import _rosenbrock


def test_rosenbrock_two_dim():
    x = np.array([1.0, 2.0])
    val = _rosenbrock(x)
    assert isinstance(val, float)
    # verify it's positive for this input
    assert val > 0.0
