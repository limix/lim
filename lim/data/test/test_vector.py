from lim.data.vector import Vector

from numpy import asarray
from numpy.testing import assert_array_equal


def test_creation():
    labels = ['sample01', 'sample02', 'sample03']
    values = [34.3, 2.3, 103.4]
    v = Vector(labels, values)

    assert_array_equal(v['sample03'], [103.4])
    assert_array_equal(v['sample01', 'sample03'], [34.3, 103.4])
    assert_array_equal(v[asarray(['sample01', 'sample03'])], [34.3, 103.4])
