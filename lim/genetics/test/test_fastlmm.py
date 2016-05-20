from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal

from ..fastlmm import FastLMM
from ...gp.regression import RegGP
from ...util.fruits import Apples
from ...cov import LinearCov
from ...cov import EyeCov
from ...cov import SumCov
from ...mean import OffsetMean
from ...func import check_grad
from ...random import GPSampler

def test_learn():
    random = np.random.RandomState(9458)
    N = 800
    X = random.randn(N, 900)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])
    offset = 1.2

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

    y = GPSampler(mean, cov).sample(random)

    flmm = FastLMM(y, X)
    flmm.learn()

    assert_almost_equal(1.2120718670, flmm.offset, decimal=6)
    assert_almost_equal(1.2979613599, flmm.genetic_variance, decimal=5)
    assert_almost_equal(1.6317660354, flmm.noise_variance, decimal=5)
