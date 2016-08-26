import numpy as np
import numpy.testing as npt

from lim.mean import OffsetMean
from lim.cov import LinearCov


def test_UniFuncWrapper_value():
    mean = OffsetMean()
    random = np.random.RandomState(0)
    n = 10
    o = 1.3
    mean.offset = o
    mean.set_data(n)
    npt.assert_almost_equal(o * np.ones(n), mean.data('learn').value())


def test_BinFuncWrapper_value():
    cov = LinearCov()
    random = np.random.RandomState(0)
    X = random.randn(10, 5)

    K = np.empty((10, 10))
    for (i, xi) in enumerate(X):
        for (j, xj) in enumerate(X):
            K[i, j] = cov.value(xi, xj)

    cov.set_data((X, X))
    npt.assert_almost_equal(K, cov.data('learn').value())


def test_UniFuncWrapper_gradient():
    mean = OffsetMean()
    random = np.random.RandomState(0)
    n = 10
    o = 1.3
    mean.offset = o
    mean.set_data(n)

    npt.assert_almost_equal(mean.gradient(n), mean.data('learn').gradient())


def test_BinFuncWrapper_gradient():
    cov = LinearCov()
    random = np.random.RandomState(0)
    X = random.randn(10, 5)

    dK = cov.gradient(X, X)[0]

    cov.set_data((X, X))
    npt.assert_almost_equal(dK, cov.data('learn').gradient()[0])
