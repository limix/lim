from numpy.random import RandomState
from numpy.testing import assert_equal

from lim.random import GLMMSampler
from lim.lik import Binomial
from lim.lik import Poisson
from lim.link import Logit
from lim.link import Log


def test_binomial_sampler():
    random = RandomState(4503)
    logit = Logit()
    binom = Binomial(5, 12, logit)
    assert_equal(binom.sample(0, random), 7)


def test_poisson_sampler():
    random = RandomState(4503)
    link = Log()
    poisson = Poisson(5, link)
    assert_equal(poisson.sample(0, random), 1)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(0, random), 2)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(-10, random), 0)
    assert_equal(poisson.sample(+5, random), 158)
