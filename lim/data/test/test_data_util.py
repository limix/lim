from numpy import array
from numpy.testing import assert_array_almost_equal

from lim.data import asarray


def test_asarray():
    a = array([[1.2, -0.1, 0.0, 5.0]])
    b = asarray(a)
    assert_array_almost_equal(a, b)
