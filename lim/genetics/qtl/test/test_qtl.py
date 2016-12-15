from __future__ import division

from numpy import sqrt
from numpy import dot
from numpy.random import RandomState
from numpy.testing import assert_allclose

from lim.tool.normalize import stdnorm
from lim.random.canonical import bernoulli
from lim.random.canonical import binomial
from lim.random.canonical import poisson
from lim.genetics.qtl import scan
from lim.genetics.phenotype import (NormalPhenotype, BernoulliPhenotype,
                                    BinomialPhenotype, PoissonPhenotype)


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

    qtl = scan(NormalPhenotype(y), X, G=G, progress=False)
    assert_allclose(
        qtl.pvalues(), [
            0.0705409218415, 1.33152093178e-13, 2.48707231047e-05,
            0.259664271433, 0.93041182258
        ],
        rtol=1e-5)

def test_qtl_normal_scan_covariate_redundance():
    random = RandomState(2)

    N = 50
    G = random.randn(N, N + 100)
    G = stdnorm(G, 0)
    G /= sqrt(G.shape[1])

    p = 5
    X = random.randn(N, p)
    X = stdnorm(X, 0)
    X /= sqrt(X.shape[1])

    u1 = random.randn(N+100) / sqrt(N+100)
    u2 = random.randn(p) / sqrt(p)

    y = dot(G, u1) + dot(X, u2)

    X[:] = 1
    qtl = scan(NormalPhenotype(y), X, G=G, progress=False)
    assert_allclose(qtl.pvalues(), [1]*p)

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

    qtl = scan(BinomialPhenotype(nsuccesses, ntrials), X, G=G, progress=False)
    assert_allclose(
        qtl.pvalues(), [
            0.0275759504728,
            0.570733489676,
            0.69020414305,
            0.757099966035,
            0.0937948997269,
        ],
        rtol=1e-2)

def test_qtl_binomial_scan_covariate_redundance():
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

    X[:] = 1
    qtl = scan(BinomialPhenotype(nsuccesses, ntrials), X, G=G, progress=False)
    assert_allclose(qtl.pvalues(), [1] * p)

def test_qtl_poisson_scan():
    random = RandomState(9)

    N = 50
    G = random.randn(N, N + 100)
    G = stdnorm(G, 0)
    G /= sqrt(G.shape[1])

    p = 5
    X = random.randn(N, p)
    X = stdnorm(X, 0)
    X /= sqrt(X.shape[1])

    noccurrences = poisson(
        -0.1,
        G,
        causal_variants=X,
        causal_variance=0.1,
        random_state=random)

    qtl = scan(PoissonPhenotype(noccurrences), X, G=G, progress=False)

    assert_allclose(
        qtl.pvalues(), [
            0.3321663377,
            0.6568225859,
            0.2170163554,
            0.1923094375,
            0.5163075213
        ],
        rtol=1e-4)

def test_qtl_bernoulli_scan():
    random = RandomState(9)

    N = 50
    G = random.randn(N, N + 100)
    G = stdnorm(G, 0)
    G /= sqrt(G.shape[1])

    p = 5
    X = random.randn(N, p)
    X = stdnorm(X, 0)
    X /= sqrt(X.shape[1])

    outcome = bernoulli(
        -0.1,
        G,
        causal_variants=X,
        causal_variance=0.1,
        random_state=random)

    qtl = scan(BernoulliPhenotype(outcome), X, G=G, progress=False)

    assert_allclose(
        qtl.pvalues(), [0.275255011086,
                        0.798092933746,
                        0.293941509957,
                        0.0454495193992,
                        0.900534408095
        ],
        rtol=1e-4)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
