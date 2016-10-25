from __future__ import division

import numpy as np
from numpy import ones
from numpy.testing import assert_almost_equal

from lim.genetics import FastLMM
from lim.util.fruits import Apples
from lim.cov import LinearCov
from lim.cov import EyeCov
from lim.cov import SumCov
from lim.mean import OffsetMean
from lim.random import RegGPSampler
from lim.random import FastLMMSampler


def test_learn():
    random = np.random.RandomState(9458)
    N = 800
    X = random.randn(N, 900)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N, purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((Apples(N), Apples(N)), purpose='sample')

    cov = SumCov([cov_left, cov_right])

    y = RegGPSampler(mean, cov).sample(random)

    flmm = FastLMM(y, ones((N, 1)), X)

    flmm.learn()

    assert_almost_equal(flmm.beta[0], 1.01207186704, decimal=6)
    assert_almost_equal(flmm.genetic_variance, 1.29796143002, decimal=5)
    assert_almost_equal(flmm.environmental_variance, 1.63176598726, decimal=5)


def test_predict_1():
    random = np.random.RandomState(228)
    N = 800
    X = random.randn(N, 900)

    offset = 1.2
    scale = 3.0
    delta = 0.5
    y = FastLMMSampler(offset, scale, delta, X).sample(random)

    flmm = FastLMM(y, ones((N, 1)), X)
    flmm.learn()
    assert_almost_equal(flmm.predict(ones((N, 1)), X).logpdf(y),
                        -1092.1273501778442, decimal=4)


def test_predict_2():
    random = np.random.RandomState(228)
    N = 200
    X = random.randn(N, 300)

    offset = 1.2
    scale = 3.0
    delta = 0.5
    y = FastLMMSampler(offset, scale, delta, X).sample(random)

    flmm = FastLMM(y, ones((N, 1)), X)
    flmm.learn()
    p = flmm.predict(ones((N, 1))[5, :], X[5, :])
    y5 = y[5]
    y6 = y[6]
    assert_almost_equal(p.logpdf(y5), -1.28820823178)
    assert_almost_equal(p.logpdf(y6), -4.28963888498)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
