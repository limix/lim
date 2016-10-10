from numpy.random import RandomState
from numpy.testing import assert_equal

from lim.random import GLMMSampler
from lim.mean import OffsetMean
from lim.cov import LinearCov
from lim.cov import EyeCov
from lim.cov import SumCov
from lim.lik import Binomial
from lim.lik import Poisson
from lim.link import LogitLink
from lim.link import LogLink
from lim.util.fruits import Apples


def test_binomial_sampler():
    random = RandomState(4503)
    link = LogitLink()
    binom = Binomial(5, 12, link)
    assert_equal(binom.sample(0, random), 7)


def test_poisson_sampler():
    random = RandomState(4503)
    link = LogLink()
    poisson = Poisson(5, link)
    assert_equal(poisson.sample(0, random), 1)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(0, random), 2)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(-10, random), 0)
    assert_equal(poisson.sample(+5, random), 158)


def test_GLMMSampler_poisson():
    random = RandomState(4503)
    X = random.randn(10, 15)
    link = LogLink()
    lik = Poisson(5, link)

    mean = OffsetMean()
    mean.offset = 1.2
    mean.set_data(10, 'sample')
    cov = LinearCov()
    cov.set_data((X, X), 'sample')
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(sampler.sample(random),
                 [0, 289, 0, 11, 0, 0, 176, 0, 228, 82])

    mean = OffsetMean()
    mean.offset = 0.0
    mean.set_data(10, 'sample')

    cov1 = LinearCov()
    cov1.set_data((X, X), 'sample')

    cov2 = EyeCov()
    a = Apples(10)
    cov2.set_data((a, a), 'sample')

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    sampler = GLMMSampler(lik, mean, cov)

    assert_equal(sampler.sample(random), [2, 0, 1, 2, 1, 1, 1, 2, 0, 0])


def test_GLMMSampler_binomial():
    random = RandomState(4503)
    X = random.randn(10, 15)
    link = LogitLink()
    lik = Binomial(3, 5, link)

    mean = OffsetMean()
    mean.offset = 1.2
    mean.set_data(10, 'sample')
    cov = LinearCov()
    cov.set_data((X, X), 'sample')
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(sampler.sample(random),
                 [0, 5, 0, 5, 1, 1, 5, 0, 5, 5])

    mean.offset = 0.
    assert_equal(sampler.sample(random), [5, 4, 1, 0, 0, 1, 4, 5, 5, 0])

    mean = OffsetMean()
    mean.offset = 0.0
    mean.set_data(10, 'sample')

    cov1 = LinearCov()
    cov1.set_data((X, X), 'sample')

    cov2 = EyeCov()
    a = Apples(10)
    cov2.set_data((a, a), 'sample')

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    lik = Binomial(3, 100, link)
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [
                 56, 56, 55, 51, 59, 45, 47, 43, 51, 38])

    cov2.scale = 100.
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [99, 93, 99, 75, 77, 0, 0, 100, 99,
                                          12])

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
