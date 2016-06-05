import os

from numpy import asarray
from numpy import loadtxt
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

import lim

def _test_2d(reader, fp, dtype):
    R = loadtxt(fp, delimiter=',', dtype=dtype)
    X = reader(fp, dtype=dtype)
    assert_array_equal(X, R)
    assert_array_equal(X[1:,:], R[1:,:])
    assert_array_equal(X[:,1:], R[:,1:])
    assert_array_equal(X[1:,1:], R[1:,1:])
    assert_array_equal(X[1:-1,:], R[1:-1,:])
    assert_array_equal(X[1:-1,-2:-1], R[1:-1,-2:-1])
    assert_equal(X.shape, R.shape)
    assert_equal(X.ndim, R.ndim)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            assert_equal(X[i,j], R[i,j])

    assert_equal(bytes(X), bytes(R))

    assert_array_equal(asarray(X[:]), R)


def test_2d():
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, 'data')
    _test_2d(lim.csvpath, os.path.join(root, '2d_array.csv'), float)
    _test_2d(lim.csvpath, os.path.join(root, '2d_array_bytes.csv'), bytes)

test_2d()
