import numpy as np

from netgraph._utils import _get_orthogonal_unit_vector


def test_get_orthogonal_unit_vector_zero_input():
    vec = np.array([[0.0, 0.0]])
    result = _get_orthogonal_unit_vector(vec)
    assert np.all(np.isfinite(result))
