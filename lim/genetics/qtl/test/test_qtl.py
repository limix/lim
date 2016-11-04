from __future__ import division

from numpy import sqrt
from numpy import dot
from numpy.random import RandomState
from numpy.testing import assert_allclose

from lim.tool.normalize import stdnorm
from lim.random.canonical import binomial
from lim.genetics.qtl import normal_scan
from lim.genetics.qtl import binomial_scan


def test_qtl_normal_scan():
    random = RandomState(2)

    N = 50
    G = random.randn(N, N + 100)
    G = stdnorm(G, 0)
    G /= sqrt(G.shape[1])

    p = 5
    X = random.randn(N, p)
    X = stdnorm(X, 0)
    X /= sqrt(X.shape[1])

    u1 = random.randn(N + 100) / sqrt(N + 100)
    u2 = random.randn(p) / sqrt(p)

    y = dot(G, u1) + dot(X, u2)

    qtl = normal_scan(y, X, G=G, progress=False)
    assert_allclose(
        qtl.pvalues(), [
            0.0705409218415, 1.33152093178e-13, 2.48707231047e-05,
            0.259664271433, 0.93041182258
        ],
        rtol=1e-5)


# def test_qtl_normal_scan_covariate_redundance():
#     random = RandomState(2)
#
#     N = 50
#     G = random.randn(N, N + 100)
#     G = stdnorm(G, 0)
#     G /= sqrt(G.shape[1])
#
#     p = 5
#     X = random.randn(N, p)
#     X = stdnorm(X, 0)
#     X /= sqrt(X.shape[1])
#
#     u1 = random.randn(N+100) / sqrt(N+100)
#     u2 = random.randn(p) / sqrt(p)
#
#     y = dot(G, u1) + dot(X, u2)
#
#     X[:] = 1
#     qtl = normal_scan(y, X, G=G, progress=False)


def test_qtl_binomial_scan():
    random = RandomState(9)

    N = 50
    G = random.randn(N, N + 100)
    G = stdnorm(G, 0)
    G /= sqrt(G.shape[1])

    p = 5
    X = random.randn(N, p)
    X = stdnorm(X, 0)
    X /= sqrt(X.shape[1])

    ntrials = random.randint(1, 50, N)
    nsuccesses = binomial(
        ntrials,
        -0.1,
        G,
        causal_variants=X,
        causal_variance=0.1,
        random_state=random)

    qtl = binomial_scan(nsuccesses, ntrials, X, G=G, progress=False)
    assert_allclose(
        qtl.pvalues(), [
            0.0275759504728,
            0.570733489676,
            0.69020414305,
            0.757099966035,
            0.0937948997269,
        ],
        rtol=1e-5)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
