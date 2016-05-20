from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal

from scipy.stats import pearsonr

from ..fastlmm import _FastLMMCore
from ..regression import RegGP
from ...util.fruits import Apples
from ...cov import LinearCov
from ...cov import EyeCov
from ...cov import SumCov
from ...mean import OffsetMean
from ...func import check_grad
from ...random import GPSampler
from ...genetics import eigen_design_matrix

def test_fastlmm_optimization_1():
    random = np.random.RandomState(9458)
    N = 200
    X = random.randn(N, 400)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])
    offset = 1.2

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N)
    mean.set_data(N, purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X))
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((Apples(N), Apples(N)))
    cov_right.set_data((Apples(N), Apples(N)), purpose='sample')

    cov = SumCov([cov_left, cov_right])

    y = GPSampler(mean, cov).sample(random)

    gp = RegGP(y, mean, cov)
    gp.learn()
    delta = cov_right.scale / (cov_left.scale + cov_right.scale)
    QS = eigen_design_matrix(X)
    flmmc = _FastLMMCore(y, QS[0][0], QS[0][1], QS[1][0])
    flmmc.delta = delta
    assert_almost_equal(gp.lml(), flmmc.lml())
    assert_almost_equal(mean.offset, flmmc.offset, decimal=6)

    # flmmc.learn()
    # flmm = FastLMM(y, X)
    # flmm.learn()
    # assert_almost_equal(gp.lml(), flmm.lml(), decimal=5)
