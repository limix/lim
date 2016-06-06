import os

import h5py

from numpy import asarray
from numpy import loadtxt
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

import lim

def _test_1d(lim_reader, numpy_reader):
    R = numpy_reader()
    X = lim_reader()
    assert_array_equal(X, R)
    assert_array_equal(X[1:], R[1:])
    assert_array_equal(X[1:-1], R[1:-1])
    assert_array_equal(X[-2:-1], R[-2:-1])
    assert_equal(X.shape, R.shape)
    assert_equal(X.ndim, R.ndim)
    for i in range(X.shape[0]):
        assert_equal(X[i], R[i])

    assert_array_equal(asarray(X[:]), R)

def _test_2d(lim_reader, numpy_reader):
    R = numpy_reader()
    X = lim_reader()
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

def test_2d():
    root = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(root, 'data')

    fp = os.path.join(root, '1d_array_col.csv')
    _test_1d(lambda: lim.csvpath(fp, float),
             lambda: loadtxt(fp, delimiter=',', dtype=float))

    fp = os.path.join(root, '1d_array_row.csv')
    _test_1d(lambda: lim.csvpath(fp, float),
             lambda: loadtxt(fp, delimiter=',', dtype=float))

    fp = os.path.join(root, '1d_array_col_bytes.csv')
    _test_1d(lambda: lim.csvpath(fp, bytes),
             lambda: loadtxt(fp, delimiter=',', dtype=bytes))

    fp = os.path.join(root, '1d_array_row_bytes.csv')
    _test_1d(lambda: lim.csvpath(fp, bytes),
             lambda: loadtxt(fp, delimiter=',', dtype=bytes))

    fp = os.path.join(root, '2d_array.csv')
    _test_2d(lambda: lim.csvpath(fp, float),
             lambda: loadtxt(fp, delimiter=',', dtype=float))

    fp = os.path.join(root, '2d_array_bytes.csv')
    _test_2d(lambda: lim.csvpath(fp, bytes),
             lambda: loadtxt(fp, delimiter=',', dtype=bytes))

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lambda: lim.h5path(fp, '/group/1d_array'),
                 lambda: f['/group/1d_array'])

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lambda: lim.h5path(fp, '/group/1d_array_bytes'),
                 lambda: f['/group/1d_array_bytes'])

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lambda: lim.h5path(fp, '/group/2d_array'),
                 lambda: f['/group/2d_array'])

    fp = os.path.join(root, 'array.h5')
    with h5py.File(fp, 'r') as f:
        _test_1d(lambda: lim.h5path(fp, '/group/2d_array_bytes'),
                 lambda: f['/group/2d_array_bytes'])
