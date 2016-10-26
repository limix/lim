from __future__ import division

from numpy import array, dot, empty, hstack, ones, sqrt
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition
from lim.inference.ep import BernoulliEP


def test_bernoulli_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    y = array([1., 0., 1.])
    ep = BernoulliEP(y, M, hstack(Q), empty((n, 0)), hstack(S) + 1.0)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0.
    assert_almost_equal(ep.lml(), -2.344936587017978)
    assert_almost_equal(ep.sigma2_epsilon, 0)
    assert_almost_equal(ep.sigma2_b, 1)


def test_bernoulli_gradient_over_v():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    y = array([1., 0., 1.])
    ep = BernoulliEP(y, M, hstack(Q), empty((n, 0)), hstack(S) + 1.0)
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


def test_bernoulli_optimize():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    M = ones((nsamples, 1))

    u = random.randn(nfeatures)

    z = 0.1 + dot(G, u) + 0.5 * random.randn(nsamples)

    y = empty(nsamples)
    y[z > 0] = 1
    y[z <= 0] = 0

    (Q, S) = qs_decomposition(G)

    ep = BernoulliEP(y, M, Q[0], Q[1], S[0])
    ep.optimize()
    assert_almost_equal(ep.lml(), -16.691454697813427)
    assert_almost_equal(ep.sigma2_b, 5551.30530403)
    assert_almost_equal(ep.sigma2_epsilon, 0.0)
    assert_almost_equal(ep.beta[0], 2.82422935655)
