from __future__ import division

import numpy as np
from numpy import dot, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from lim.genetics.phenotype import (BernoulliPhenotype, BinomialPhenotype,
                                    NormalPhenotype, PoissonPhenotype)
from lim.genetics.qtl import scan
from lim.random.canonical import bernoulli, binomial, poisson
from lim.tool.normalize import stdnorm


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

    u1 = random.randn(N + 100) / sqrt(N + 100)
    u2 = random.randn(p) / sqrt(p)

    y = dot(G, u1) + dot(X, u2)

    X[:] = 1
    qtl = scan(NormalPhenotype(y), X, G=G, progress=False)
    assert_allclose(qtl.pvalues(), [1] * p)


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
            0.024850888252989, 0.569692542168291, 0.6901692002748263,
            0.7570352252604872, 0.0888544623627765
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
    assert_allclose(qtl.pvalues(), [1] * p, rtol=1e-4)


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
        -0.1, G, causal_variants=X, causal_variance=0.1, random_state=random)

    qtl = scan(PoissonPhenotype(noccurrences), X, G=G, progress=False)

    assert_allclose(
        qtl.pvalues(), [
            0.3602609452849577, 0.5858386291857423, 0.2578859083822661,
            0.2583832367791912, 0.3712543845933317
        ],
        rtol=1e-2)


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
        -0.1, G, causal_variants=X, causal_variance=0.1, random_state=random)

    qtl = scan(BernoulliPhenotype(outcome), X, G=G, progress=False)

    assert_allclose(
        qtl.pvalues(), [
            0.2746075778074198, 0.795602783193831, 0.27632759390398,
            0.0409249249070836, 0.8975492615275441
        ],
        rtol=1e-4)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
