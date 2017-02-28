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
            0.0204700427237634, 0.5696100459850757, 0.6900110532522933,
            0.7514419325181403, 0.0888395185085836
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
            0.323780186575525, 0.6535310373144236, 0.2103207985450398,
            0.1849561475202371, 0.4953569917862007
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
            0.2743931472870954, 0.7925543535174397, 0.2762730442807277,
            0.0408349293752539, 0.8994393017110111
        ],
        rtol=1e-4)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
