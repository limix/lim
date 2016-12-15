from __future__ import division

from numpy.random import RandomState
from numpy.testing import (assert_equal, assert_array_less)

from lim.random import GLMMSampler
from lim.random.canonical import bernoulli
from lim.random.canonical import binomial
from lim.random.canonical import poisson
from lim.util.fruits import Apples

def test_canonical_bernoulli_sampler():
    random = RandomState(9)
    G = random.randn(10, 5)

    y = bernoulli(0.1, G, random_state=random)
    assert_array_less(y, [2] * 10)

def test_canonical_binomial_sampler():
    random = RandomState(9)
    G = random.randn(10, 5)

    y = binomial(5, 0.1, G, random_state=random)
    assert_array_less(y, [5 + 1] * 10)

    ntrials = [2, 3, 1, 1, 4, 5, 1, 2, 1, 1]
    y = binomial(ntrials, -0.1, G, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])

def test_canonical_poisson_sampler():
    random = RandomState(9)
    G = random.randn(10, 5)

    y = poisson(0.1, G, random_state=random)
    assert_array_less(y, [20] * len(y))


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
