import numpy as np
import numpy.testing as npt

from ..data import set_data
from ...mean import OffsetMean
from ...cov import LinearCov

def test_UniFuncWrapper_value():
    mean = OffsetMean()
    random = np.random.RandomState(0)
    n = 10
    o = 1.3
    mean.offset = o
    set_data(mean, n)
    npt.assert_almost_equal(o*np.ones(n), mean.learn.value())

def test_BinFuncWrapper_value():
    cov = LinearCov()
    random = np.random.RandomState(0)
    X = random.randn(10, 5)

    K = np.empty((10, 10))
    for (i, xi) in enumerate(X):
        for (j, xj) in enumerate(X):
            K[i, j] = cov.value(xi, xj)

    set_data(cov, X)
    npt.assert_almost_equal(K, cov.learn.value())

def test_UniFuncWrapper_gradient():
    mean = OffsetMean()
    random = np.random.RandomState(0)
    n = 10
    o = 1.3
    mean.offset = o
    set_data(mean, n)

    npt.assert_almost_equal(mean.gradient(n), mean.learn.gradient())

def test_BinFuncWrapper_gradient():
    cov = LinearCov()
    random = np.random.RandomState(0)
    X = random.randn(10, 5)

    dK = cov.gradient(X, X)[0]

    set_data(cov, X)
    npt.assert_almost_equal(dK, cov.learn.gradient()[0])
