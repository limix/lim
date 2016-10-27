from __future__ import division

from numpy import array, dot, empty, hstack, ones, pi, sqrt, zeros, exp
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from limix_math.linalg import qs_decomposition
from lim.inference.ep import PoissonEP


def test_poisson_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S) = qs_decomposition(G)
    noccurrences = array([1., 0., 5.])
    ep = PoissonEP(noccurrences, M, hstack(Q),
                   empty((n, 0)), hstack(S) + 1.0)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0
    assert_almost_equal(ep.lml(), -6.54382414158)


def test_poisson_optimize():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    y = zeros(nsamples)
    for i in range(nsamples):
        y[i] = random.poisson(lam=exp(z[i]))
    (Q, S) = qs_decomposition(G)

    M = ones((nsamples, 1))
    ep = PoissonEP(y, M, Q[0], Q[1], S[0])
    ep.optimize()
    assert_almost_equal(ep.lml(), -77.90850467833714, decimal=3)
    assert_almost_equal(ep.sigma2_b, 3.38637577198, decimal=1)
    assert_almost_equal(ep.sigma2_epsilon, 0.858399432528, decimal=1)
    assert_almost_equal(ep.beta[0], 0.314709077094, decimal=1)
    assert_almost_equal(ep.heritability, 0.797775054939, decimal=1)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
