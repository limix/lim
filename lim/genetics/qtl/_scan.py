"""Quantitative trait locus discovery."""

from __future__ import absolute_import

import logging

from numpy import sqrt
from numpy import ones
from numpy import empty_like
from numpy import copyto

from numpy_sugar.linalg import (economic_qs, economic_qs_linear)
from numpy_sugar import is_all_finite

from ._qtl import QTLScan
from ..background import Background
from ...tool.kinship import gower_normalization
from ...tool.normalize import stdnorm

def scan(phenotype, X, G=None, K=None, covariates=None, progress=True):
    """Association between genetic variants and phenotype.

    Matrix `X` shall contain the genetic markers (e.g., number of minor
    alleles) with rows and columns representing samples and genetic markers,
    respectively.

    The user must specify only one of the parameters `G` and `K` for defining
    the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates,
    :math:`P_c` the number of genetic markers to be tested, and :math:`P_b`
    the number of genetic markers used for Kinship estimation.

    Args:
        y          (array_like): Phenotype. Dimension (:math:`N\\times 0`).
        X          (array_like): Candidate genetic markers (or any other
                                 type of explanatory variable) whose
                                 association with the phenotype will be
                                 tested. Dimension (:math:`N\\times P_c`).
        G          (array_like): Genetic markers matrix used internally for
                                 kinship estimation. Dimension
                                 (:math:`N\\times P_b`).
        K          (array_like): Kinship matrix. Dimension
                                 (:math:`N\\times N`).
        covariates (array_like): Covariates. Default is an offset.
                                 Dimension (:math:`N\\times S`).
        progress    (bool)     : Shows progress. Defaults to `True`.

    Returns:
        A :class:`lim.genetics.qtl._canonical.CanonicalLRTScan` instance.
    """
    logger = logging.getLogger(__name__)
    logger.info('%s association scan has started.', phenotype.likelihood_name)

    n = phenotype.sample_size
    covariates = ones((n, 1)) if covariates is None else covariates

    X = _clone(X)
    G = _clone(G)
    K = _clone(K)

    if not is_all_finite(X):
        raise ValueError("The candidate matrix X has non-finite values.")

    if G is not None and not is_all_finite(G):
        raise ValueError("The genetic markers matrix G has non-finite values.")

    if K is not None and not is_all_finite(K):
        raise ValueError("The Kinship matrix K has non-finite values.")

    background = Background()

    (Q0, Q1, S0) = _genetic_preprocess(X, G, K, background)
    qtl = QTLScan(phenotype, covariates, X, Q0, Q1, S0)
    qtl.progress = progress
    qtl.compute_statistics()

    return qtl

def _genetic_preprocess(X, G, K, background):
    logger = logging.getLogger(__name__)
    logger.info("Number of candidate markers to scan: %d", X.shape[1])

    if K is not None:
        background.provided_via_variants = False
        logger.info('Covariace matrix normalization.')
        gower_normalization(K, out=K)

    if G is not None:
        background.provided_via_variants = True
        background.nvariants = G.shape[1]
        background.constant_nvariants = sum(G.std(0) == 0)

        logger.info('Genetic markers normalization.')
        stdnorm(G, 0, out=G)
        G /= sqrt(G.shape[1])

    if G is None and K is None:
        raise Exception('G and K cannot be both None.')

    logger.info('Computing the economic eigen decomposition.')
    if K is None:
        QS = economic_qs_linear(G)
    else:
        QS = economic_qs(K)

    Q0, Q1 = QS[0]
    S0 = QS[1]

    background.background_rank = len(S0)

    logger.info('Genetic marker candidates normalization.')
    stdnorm(X, 0, out=X)
    X /= sqrt(X.shape[1])

    return (Q0, Q1, S0)


def _clone(X):
    if X is None:
        return None
    Y = empty_like(X, dtype=float, order='C')
    copyto(Y, X)
    return Y
