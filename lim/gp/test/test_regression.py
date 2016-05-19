from __future__ import division

import numpy as np
import numpy.testing as npt

from scipy.stats import pearsonr

from ..regression import RegGP
from ...cov import LinearCov
from ...mean import OffsetMean
from ...data import set_data
from ...func import check_grad
from ...random import GPSampler

def test_regression_value():
    random = np.random.RandomState(94584)
    N = 400
    X = random.randn(N, 500)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    set_data(mean, N)

    cov = LinearCov()
    cov.scale = 1.0
    set_data(cov, X)

    y = random.randn(N)

    gp = RegGP(y, mean, cov)
    npt.assert_almost_equal(gp.lml(), -1495.12790401)

def test_regression_gradient():
    random = np.random.RandomState(94584)
    N = 400
    X = random.randn(N, 500)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    set_data(mean, N)

    cov = LinearCov()
    cov.scale = 1.0
    set_data(cov, X)

    y = random.randn(N)

    gp = RegGP(y, mean, cov)
    g = gp.lml_gradient()

    def func(x):
        cov.scale = np.exp(x[0])
        return gp.lml()

    def grad(x):
        cov.scale = np.exp(x[0])
        return gp.lml_gradient()

    npt.assert_almost_equal(check_grad(func, grad, [0]), 0)

def test_maximize_1():
    random = np.random.RandomState(94584)
    N = 400
    X = random.randn(N, 500)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    set_data(mean, N)

    cov = LinearCov()
    cov.scale = 1.0
    set_data(cov, X)

    y = random.randn(N)

    gp = RegGP(y, mean, cov)

    gp.learn()
    npt.assert_almost_equal(gp.lml(), -805.453722549)

def test_maximize_2():
    random = np.random.RandomState(94584)
    N = 400
    X = random.randn(N, 500)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    set_data(mean, N)

    cov = LinearCov()
    cov.scale = 1.0
    set_data(cov, X)

    y = random.randn(N)

    gp = RegGP(y, mean, cov)

    gp.learn()
    npt.assert_almost_equal(gp.lml(), -761.517250775)

def test_predict():
    random = np.random.RandomState(94584)
    N = 400
    nlearn = N - N//5
    npred = N//5
    X = random.randn(N, 500)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    set_data(mean, N, purpose='sample')
    set_data(mean, nlearn, purpose='learn')
    set_data(mean, npred, purpose='predict')

    cov = LinearCov()
    cov.scale = 1.0
    set_data(cov, X, purpose='sample')
    set_data(cov, X[:nlearn, :], purpose='learn')
    set_data(cov, X[-npred:, :], purpose='predict')

    y = GPSampler(mean, cov).sample(random)

    gp = RegGP(y[:nlearn], mean, cov)

    gp.learn()
    npt.assert_almost_equal(gp.lml(), -1377.9245876374036)

    ypred = gp.predict()
    (corr, pval) = pearsonr(y[-npred:], ypred)
    
    npt.assert_almost_equal(corr, 0.81494179361621788)
    npt.assert_almost_equal(pval, 0)
