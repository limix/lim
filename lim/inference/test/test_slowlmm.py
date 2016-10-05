from __future__ import division

from numpy import exp
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from lim.inference import SlowLMM
from lim.func import check_grad
from lim.cov import LinearCov
from lim.mean import OffsetMean


def test_slowlmm_value():
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

# def test_predict_1():
#     random = np.random.RandomState(94584)
#     N = 400
#     nlearn = N - N//5
#     npred = N//5
#     X = random.randn(N, 500)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.set_data(N, purpose='sample')
#     mean.set_data(nlearn, purpose='learn')
#     mean.set_data(npred, purpose='predict')
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X), purpose='sample')
#     cov.set_data((X[:nlearn, :], X[:nlearn, :]), purpose='learn')
#     cov.set_data((X[:nlearn, :], X[-npred:, :]), purpose='learn_predict')
#     cov.set_data((X[-npred:, :], X[-npred:, :]), purpose='predict')
#
#     y = RegGPSampler(mean, cov).sample(random)
#
#     gp = RegGP(y[:nlearn], mean, cov)
#
#     gp.learn()
#     assert_almost_equal(gp.lml(), -1377.9245876374036)
#
#     pred = gp.predict()
#     (corr, pval) = pearsonr(y[-npred:], pred.mean)
#
#     assert_almost_equal(corr, 0.81494179361621788)
#     assert_almost_equal(pval, 0)
#
# def test_predict_2():
#     random = np.random.RandomState(94584)
#     N = 400
#     nlearn = N - N//5
#     npred = N//5
#     X = random.randn(N, 500)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.set_data(N, purpose='sample')
#     mean.set_data(nlearn, purpose='learn')
#     mean.set_data(npred, purpose='predict')
#
#     cov_left = LinearCov()
#     cov_left.scale = 1.0
#     cov_left.set_data((X, X), purpose='sample')
#     cov_left.set_data((X[:nlearn, :], X[:nlearn, :]), purpose='learn')
#     cov_left.set_data((X[:nlearn, :], X[-npred:, :]), purpose='learn_predict')
#     cov_left.set_data((X[-npred:, :], X[-npred:, :]), purpose='predict')
#
#     cov_right = EyeCov()
#     cov_right.scale = 0.2
#     cov_right.set_data((Apples(N), Apples(N)), purpose='sample')
#     cov_right.set_data((Apples(nlearn), Apples(nlearn)), purpose='learn')
#     cov_right.set_data((Apples(nlearn), Apples(npred)), purpose='learn_predict')
#     cov_right.set_data((Apples(npred), Apples(npred)), purpose='predict')
#
#     cov = SumCov([cov_left, cov_right])
#
#     y = RegGPSampler(mean, cov).sample(random)
#
#     gp = RegGP(y[:nlearn], mean, cov)
#
#     cov_left.scale = 0.1
#     cov_right.scale = 5.0
#     gp.learn()
#     assert_almost_equal(gp.lml(), -1379.03320103)
#
#     pred = gp.predict()
#
#     (corr, pval) = pearsonr(y[-npred:], pred.mean)
#
#     assert_almost_equal(corr, 0.81700837814)
#     assert_almost_equal(pval, 0)
#
# def test_predict_3():
#     random = np.random.RandomState(1)
#     N = 400
#     nlearn = N - N//5
#     npred = N//5
#     X = random.randn(N, 500)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.set_data(N, purpose='sample')
#     mean.set_data(nlearn, purpose='learn')
#     mean.set_data(npred, purpose='predict')
#
#     cov_left = LinearCov()
#     cov_left.scale = 1.2
#     cov_left.set_data((X, X), purpose='sample')
#     cov_left.set_data((X[:nlearn, :], X[:nlearn, :]), purpose='learn')
#     cov_left.set_data((X[:nlearn, :], X[-npred:, :]), purpose='learn_predict')
#     cov_left.set_data((X[-npred:, :], X[-npred:, :]), purpose='predict')
#
#     cov_right = EyeCov()
#     cov_right.scale = 0.1
#     cov_right.set_data((Apples(N), Apples(N)), purpose='sample')
#     cov_right.set_data((Apples(nlearn), Apples(nlearn)), purpose='learn')
#     cov_right.set_data((Apples(nlearn), Apples(npred)), purpose='learn_predict')
#     cov_right.set_data((Apples(npred), Apples(npred)), purpose='predict')
#
#     cov = SumCov([cov_left, cov_right])
#
#     y = RegGPSampler(mean, cov).sample(random)
#
#     gp = RegGP(y[:nlearn], mean, cov)
#
#     cov_left.scale = 0.1
#     cov_right.scale = 5.0
#
#     gp.learn()
#     pred = gp.predict()
#     assert_almost_equal(pred.logpdf(y[-npred:]), -316.470733059)
