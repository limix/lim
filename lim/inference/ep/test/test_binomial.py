from __future__ import division

from numpy import array, dot, empty, hstack, ones, pi, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition
from lim.inference.ep import BinomialEP


def test_binomial_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    ep = BinomialEP(nsuccesses, ntrials, M, hstack(Q),
                    empty((n, 0)), hstack(S) + 1.0)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0
    assert_almost_equal(ep.lml(), -2.344936587017978)


def test_binomial_gradient_over_v():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    ep = BinomialEP(nsuccesses, ntrials, M, hstack(Q),
                    empty((n, 0)), hstack(S) + 1.0)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0.

    analytical_gradient = ep._gradient_over_v()

    lml0 = ep.lml()
    step = 1e-5
    ep.v = ep.v + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_almost_equal(empirical_gradient, analytical_gradient, decimal=4)

    ep.v = 0.5
    ep.delta = 0.0

    analytical_gradient = ep._gradient_over_v()

    lml0 = ep.lml()
    step = 1e-5
    ep.v = ep.v + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_almost_equal(empirical_gradient, analytical_gradient, decimal=4)


def test_binomial_gradient_over_both():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    ep = BinomialEP(nsuccesses, ntrials, M, hstack(Q),
                    empty((n, 0)), hstack(S) + 1.0)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.v = 1.5
    ep.delta = 0.3

    analytical_gradient = ep._gradient_over_both()[0]

    lml0 = ep.lml()
    step = 1e-5
    ep.v = ep.v + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_almost_equal(empirical_gradient, analytical_gradient, decimal=4)

    ep.v = 1.5
    ep.delta = 0.3

    analytical_gradient = ep._gradient_over_both()[1]

    lml0 = ep.lml()
    step = 1e-5
    ep.delta = ep.delta + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_almost_equal(empirical_gradient, analytical_gradient, decimal=4)


def test_binomial_optimize():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    ntrials = random.randint(10, 500, size=nsamples)

    y = zeros(nsamples)
    for i in range(len(ntrials)):
        y[i] = sum(z[i] + random.logistic(scale=pi / sqrt(3),
                                          size=ntrials[i]) > 0)
    (Q, S) = qs_decomposition(G)

    M = ones((nsamples, 1))
    ep = BinomialEP(y, ntrials, M, Q[0], Q[1], S[0])
    ep.optimize()

    assert_almost_equal(ep.lml(), -144.23818408071736, decimal=3)
    assert_almost_equal(ep.sigma2_b, 1.13419722224, decimal=4)
    assert_almost_equal(ep.sigma2_epsilon, 0.122855059169, decimal=4)
    assert_almost_equal(ep.beta[0], 0.110667132173, decimal=4)
    assert_almost_equal(ep.heritability, 0.902267343224, decimal=4)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
