import os

import h5py

from numpy import array
from numpy import asarray
from numpy import loadtxt
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

import lim

def _test_1d(X, R):
    assert_array_equal(X, R)
    assert_array_equal(X[1:], R[1:])
    assert_array_equal(X[1:-1], R[1:-1])
    assert_array_equal(X[-2:-1], R[-2:-1])
    assert_equal(X.shape, R.shape)
    assert_equal(X.ndim, R.ndim)
    for i in range(X.shape[0]):
        assert_equal(X[i], R[i])

    assert_array_equal(asarray(X[:]), R)

def _test_2d(X, R):
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

    assert_equal(bytes(X), bytes(R[:]))

    assert_array_equal(asarray(X[:]), R)

def test_arrays():
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, 'data')

    fp = os.path.join(root, '1d_array_col.csv')
    _test_1d(lim.csvpath(fp, float),
             loadtxt(fp, delimiter=',', dtype=float))

    fp = os.path.join(root, '1d_array_row.csv')
    _test_1d(lim.csvpath(fp, float),
             loadtxt(fp, delimiter=',', dtype=float))

    fp = os.path.join(root, '1d_array_col_bytes.csv')
    _test_1d(lim.csvpath(fp, bytes),
             loadtxt(fp, delimiter=',', dtype=bytes))

    fp = os.path.join(root, '1d_array_row_bytes.csv')
    _test_1d(lim.csvpath(fp, bytes),
             loadtxt(fp, delimiter=',', dtype=bytes))

    fp = os.path.join(root, '2d_array.csv')
    _test_2d(lim.csvpath(fp, float),
             loadtxt(fp, delimiter=',', dtype=float))

    fp = os.path.join(root, '2d_array_bytes.csv')
    _test_2d(lim.csvpath(fp, bytes),
             loadtxt(fp, delimiter=',', dtype=bytes))

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lim.h5path(fp, '/group/1d_array'),
                 f['/group/1d_array'])

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lim.h5path(fp, '/group/1d_array_bytes'),
                 f['/group/1d_array_bytes'])

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lim.h5path(fp, '/group/2d_array'),
                 f['/group/2d_array'])

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lim.h5path(fp, '/group/2d_array_bytes'),
                 f['/group/2d_array_bytes'])

    basepath = "/Users/horta/workspace/lim/lim/data/cplink/example/test"
    p = lim.plinkpaths(basepath)
    R = array([[0, 3, 2, 3, 3, 3], [3, 2, 1, 3, 3, 3], [3, 1, 1, 2, 2, 0]])
    X, individuals, markers = p[0], p[1], p[2]
    _test_2d(X, R)
