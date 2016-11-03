from __future__ import division

from numpy import sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from lim.tool.normalize import stdnorm
from lim.random.canonical import binomial
from lim.genetics.qtl import binomial_scan


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
    assert_allclose(qtl.pvalues(), [
        0.0275759504728,
        0.570733489676,
        0.69020414305,
        0.757099966035,
        0.0937948997269,
    ])


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
