from lim.data.column import Column

from numpy import asarray
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal


def test_creation():
    labels = ['sample01', 'sample02', 'sample03']
    values = [34.3, 2.3, 103.4]
    c = Column('sample_id', labels, values)

    assert_array_equal(c['sample03'], [103.4])
    assert_array_equal(c['sample01', 'sample03'], [34.3, 103.4])
    assert_array_equal(c[asarray(['sample01', 'sample03'])], [34.3, 103.4])
    assert_equal(c.name, 'sample_id')
