from __future__ import division

from numpy import exp
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from lim.inference import SlowLMM
from optimix import check_grad
from optimix import as_data_function
from lim.cov import LinearCov
from lim.cov import SumCov
from lim.mean import OffsetMean
from lim.mean import LinearMean


def test_slowlmm_value_1():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    assert_almost_equal(lmm.lml(), -153.623791551399108)


def test_slowlmm_value_2():
    random = RandomState(94584)
    N = 50
    X1 = random.randn(N, 3)
    X2 = random.randn(N, 100)

    mean = LinearMean(3)
    mean.set_data(X1)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X2, X2))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    assert_almost_equal(lmm.lml(), -153.091074766)

    mean.effsizes = [3.4, 1.11, -6.1]
    assert_almost_equal(lmm.lml(), -178.273116338)


def test_regression_gradient():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)

    def func(x):
        cov.scale = exp(x[0])
        return lmm.value(as_data_function(mean).value(),
                         as_data_function(cov).value())

    def grad(x):
        cov.scale = exp(x[0])
        return lmm.gradient(as_data_function(mean).value(),
                            as_data_function(cov).value(),
                            as_data_function(mean).gradient(),
                            as_data_function(cov).gradient())

    assert_almost_equal(check_grad(func, grad, [0]), 0)


def test_maximize_1():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    m = as_data_function(mean).value()
    K = as_data_function(cov).value()
    assert_almost_equal(lmm.value(m, K), -153.62379155139911)


    lmm.learn()
    m = as_data_function(mean).value()
    K = as_data_function(cov).value()
    assert_almost_equal(lmm.value(m, K), -79.899212241487518)
    assert_almost_equal(as_data_function(lmm).value(), -79.899212241487518)


def test_maximize_2():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    lmm.learn()
    assert_almost_equal(as_data_function(lmm).value(), -79.365136339619610)

def test_maximize_3():
    random = RandomState(94584)
    N = 50

    X1 = random.randn(N, 3)
    X2 = random.randn(N, 100)

    mean = LinearMean(3)
    mean.set_data(X1)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X2, X2))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)

    lmm.learn()
    assert_almost_equal(as_data_function(lmm).value(), -73.5638040543)


def test_maximize_4():
    random = RandomState(94584)
    N = 50

    X1 = random.randn(N, 3)
    X2 = random.randn(N, 100)
    X3 = random.randn(N, 50)

    mean = LinearMean(3)
    mean.set_data(X1)

    cov1 = LinearCov()
    cov1.scale = 1.0
    cov1.set_data((X2, X2))

    cov2 = LinearCov()
    cov2.scale = 0.5
    cov2.set_data((X3, X3))

    cov = SumCov([cov1, cov2])

    assert_almost_equal(as_data_function(cov).value(),
                        1.0 * X2.dot(X2.T) + 0.5 * X3.dot(X3.T))
