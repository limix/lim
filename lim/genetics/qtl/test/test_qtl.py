from __future__ import division

from numpy import sqrt
from numpy import ones
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from lim.util.fruits import Apples
from lim.cov import LinearCov
from lim.cov import EyeCov
from lim.cov import SumCov
from lim.mean import OffsetMean
from lim.random import RegGPSampler
from lim.genetics.qtl import normal_scan
from lim.genetics.qtl import normal_scan


def test_qtl_normal_scan():
    random = RandomState(9458)
    N = 800
    X = random.randn(N, 900)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
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

    y = RegGPSampler(mean, cov).sample(random)

    lrt = normal_scan(y, X=X, G=X, covariates=ones((N, 1)))

    null_model = lrt.null_model()
    print(null_model)
    effsizes = lrt.candidate_effect_sizes()
    assert_almost_equal(effsizes[0], 0.00881056802261)
    assert_almost_equal(null_model.heritability, 0.443031454528759)
    assert_almost_equal(null_model.total_variance, 2.929727405984385)
