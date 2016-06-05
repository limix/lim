import os

from numpy.testing import assert_equal

import lim

def test_h5():
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, 'data', 'horta1')
    fp = os.path.join(root, 'trait.hdf5')
    X = lim.h5path(fp, '/group1/trait1/sample_id')
    assert_equal(X.dtype, bytes)
    assert_equal(X.item(0), "sample_id1")
    assert_equal(X[1], "sample_id2")
    X = X[1:]
    assert_equal(str(X), "['sample_id2' 'sample_id3' 'sample_id4']")
    x = X[2]
    assert_equal(x, 'sample_id4')
    assert_equal(X[-1], 'sample_id4')
    X = X[:-1]
    assert_equal(X[-1].item(0), 'sample_id3')
    assert_equal(X.ndim, 1)
    assert_equal(X[-1].ndim, 0)
    assert_equal(X.shape, (2,))
