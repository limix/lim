from __future__ import division

from numpy.random import RandomState
from numpy.testing import assert_allclose

from lim.genetics.heritability import estimate
from lim.random.canonical import bernoulli as bernoulli_sampler

from lim.genetics.phenotype import BernoulliPhenotype, BinomialPhenotype
from lim.random.canonical import binomial as binomial_sampler


def test_heritability_bernoulli_estimate():
    random = RandomState(1)
    N = 50
    X = random.randn(N, N+1)
    y = bernoulli_sampler(0.0, X, random_state=random)
    assert_allclose(estimate(BernoulliPhenotype(y), X),
                    3.705859167105969e-07, rtol=1e-4, atol=1e-4)


def test_heritability_binomial_estimate():
    random = RandomState(1)
    N = 50
    X = random.randn(N, N+1)
    ntrials = random.randint(1, 100, size=N)
    y = binomial_sampler(ntrials, 0.1, X, random_state=random)
    assert_allclose(estimate(BinomialPhenotype(y, ntrials), X),
                    0.3759481268916276, rtol=1e-4)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
