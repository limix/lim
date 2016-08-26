from numpy import object_
from numpy import nan
from numpy import array
from numpy import logical_not as lnot

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from pandas import isnull

from lim.data.table import Table
from lim.data.column import Column


# def test_adding_columns():
#     t = Table()
#
#     labels = ['sample01', 'sample02', 'sample03']
#     values = [34.3, 2.3, 103.4]
#     c = Column('height', labels, values)
#     t.add(c)
#
#     labels = ['sample02', 'sample03']
#     values = ['doce', 'cogumelo']
#     c = Column('comida', labels, values)
#     t.add(c)
#
#     t.index_name = 'sample_id'
#
#     assert_array_equal(t.as_matrix()[:, 0], array([2.3, 103.4, 34.3]))
#
#     v = t.as_matrix()[:, 1]
#     v = v[lnot(isnull(v))]
#
#     r = array(['doce', 'cogumelo', nan], dtype=object_)
#     r = r[lnot(isnull(r))]
#
#     assert_array_equal(v, r)


def test_column_indexing():

    t = Table()

    labels = ['sample01', 'sample02', 'sample03']
    values = [34.3, 2.3, 103.4]
    c = Column('height', labels, values)
    t.add(c)

    labels = ['sample02', 'sample03']
    values = ['doce', 'cogumelo']
    c = Column('comida', labels, values)
    t.add(c)

    t.index_name = 'sample_id'

    c = t['comida']

    assert_equal(c['sample03'], array(['cogumelo']))
    with assert_raises(KeyError):
        print(c['sample05'])


def test_get_table_by_columns():

    t = Table()

    labels = ['sample01', 'sample02', 'sample03']
    values = [34.3, 2.3, 103.4]
    c = Column('height', labels, values)
    t.add(c)

    labels = ['sample02', 'sample03']
    values = ['doce', 'cogumelo']
    c = Column('comida', labels, values)
    t.add(c)

    t.index_name = 'sample_id'

    c = t['comida']

    assert_equal(c['sample03'], array(['cogumelo']))
    with assert_raises(KeyError):
        print(c['sample05'])
