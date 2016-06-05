import os

from numpy.testing import assert_equal

import lim

def test_csv():
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, 'data', 'horta1')

    fp = os.path.join(root, 'genotype_array.csv')

    X = lim.csvpath(fp)
    assert_equal(X.dtype, int)
    assert_equal(X[0,0], 0)
