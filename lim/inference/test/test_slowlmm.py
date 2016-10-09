from __future__ import division

from numpy import exp
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from lim.inference import SlowLMM
from lim.func import check_grad
from lim.cov import LinearCov
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
        return lmm.lml()

    def grad(x):
        cov.scale = exp(x[0])
        return lmm.lml_gradient()

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

    lmm.learn()
    assert_almost_equal(lmm.lml(), -79.899212241487504)


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
    assert_almost_equal(lmm.lml(), -79.365136339619610)


# def test_maximize_3():
#     random = RandomState(94584)
#     N = 50
#     X = random.randn(N, 100)
#
#     mean = LinearMean(3)
#     mean.offset = offset
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     lmm = SlowLMM(y, mean, cov)
#
#     lmm.learn()
#     assert_almost_equal(lmm.lml(), -79.365136339619610)
