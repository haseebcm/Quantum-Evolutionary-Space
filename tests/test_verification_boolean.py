import numpy as np
import pytest

from qes import verification


def test_coerce_numeric_array_rejects_boolean_array():
    arr = np.array([True, False, True])
    with pytest.raises(TypeError, match="must not be boolean"):
        verification._coerce_numeric_array(arr, name="candidate_state")
