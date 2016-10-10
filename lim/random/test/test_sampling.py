from numpy.random import RandomState
from numpy.testing import assert_equal

from lim.random import GLMMSampler
from lim.mean import OffsetMean
from lim.cov import LinearCov
from lim.lik import Binomial
from lim.lik import Poisson
from lim.link import LogLinkitLink
from lim.link import LogLink


def test_binomial_sampler():
    random = RandomState(4503)
    link = LogLinkitLink()
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


def test_GLMMSampler():
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

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
