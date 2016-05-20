from __future__ import division

import numpy as np
import numpy.testing as npt

from scipy.stats import pearsonr

from ..fastlmm import FastLMM
from ..regression import RegGP
from ...util.fruits import Apples
from ...cov import LinearCov
from ...cov import EyeCov
from ...cov import SumCov
from ...mean import OffsetMean
from ..fastlmm import FastLMM
from ...func import check_grad
from ...random import GPSampler

def test_regression_value():
    random = np.random.RandomState(94584)
    N = 400
    X = random.randn(N, 500)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])
    offset = 1.2

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N)

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X))

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((Apples(N), Apples(N)))

    cov = SumCov([cov_left, cov_right])

    y = random.randn(N)

    gp = RegGP(y, mean, cov)
    lml0 = gp.lml()

    flmm = FastLMM(y, X)
    flmm.offset = offset
    flmm.scale = 1.5
    flmm.delta = 1.0
    lml1 = flmm.lml()

    npt.assert_almost_equal(lml0, lml1)

    cov_left.fix('logscale')
    cov_right.fix('logscale')
    gp.learn()
    npt.assert_almost_equal(flmm.optimal_offset(), mean.offset)

# def test_regression_gradient():
#     random = np.random.RandomState(94584)
#     N = 400
#     X = random.randn(N, 500)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.fix('offset')
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     gp = RegGP(y, mean, cov)
#     g = gp.lml_gradient()
#
#     def func(x):
#         cov.scale = np.exp(x[0])
#         return gp.lml()
#
#     def grad(x):
#         cov.scale = np.exp(x[0])
#         return gp.lml_gradient()
#
#     npt.assert_almost_equal(check_grad(func, grad, [0]), 0)
#
# def test_maximize_1():
#     random = np.random.RandomState(94584)
#     N = 400
#     X = random.randn(N, 500)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.fix('offset')
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     gp = RegGP(y, mean, cov)
#
#     gp.learn()
#     npt.assert_almost_equal(gp.lml(), -805.453722549)
#
# def test_maximize_2():
#     random = np.random.RandomState(94584)
#     N = 400
#     X = random.randn(N, 500)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     gp = RegGP(y, mean, cov)
#
#     gp.learn()
#     npt.assert_almost_equal(gp.lml(), -761.517250775)
#
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
#     y = GPSampler(mean, cov).sample(random)
#
#     gp = RegGP(y[:nlearn], mean, cov)
#
#     gp.learn()
#     npt.assert_almost_equal(gp.lml(), -1377.9245876374036)
#
#     ypred = gp.predict()
#     (corr, pval) = pearsonr(y[-npred:], ypred)
#
#     npt.assert_almost_equal(corr, 0.81494179361621788)
#     npt.assert_almost_equal(pval, 0)
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
#     y = GPSampler(mean, cov).sample(random)
#
#     gp = RegGP(y[:nlearn], mean, cov)
#
#     cov_left.scale = 0.1
#     cov_right.scale = 5.0
#     gp.learn()
#     npt.assert_almost_equal(gp.lml(), -1379.03320103)
#
#     ypred = gp.predict()
#     (corr, pval) = pearsonr(y[-npred:], ypred)
#
#     npt.assert_almost_equal(corr, 0.81700837814)
#     npt.assert_almost_equal(pval, 0)
