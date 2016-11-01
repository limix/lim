"""Quantitative trait locus discovery."""

from __future__ import absolute_import

import logging

from tabulate import tabulate

from numpy import ascontiguousarray
from numpy import sqrt

from limix_math import (economic_qs, economic_qs_linear)

from ..phenotype import NormalPhenotype
from ..phenotype import BinomialPhenotype
from ..background import Background
from ...tool.kinship import gower_normalization
from ...tool.normalize import stdnorm


class InputInfo(object):
    def __init__(self):
        self.background_markers_user_provided = None
        self.nconst_background_markers = None
        self.covariates_user_provided = None
        self.nconst_markers = None
        self.kinship_rank = None
        self.candidate_nmarkers = None
        self.phenotype_info = None
        self.background_nmarkers = None

        self.effective_X = None
        self.effective_G = None
        self.effective_K = None

        self.S = None
        self.Q = None

    def __str__(self):
        t = []
        t += self.phenotype_info.get_info()
        if self.background_markers_user_provided:
            t.append(['Background data', 'provided via markers'])
            t.append(['# background markers', self.background_nmarkers])
            t.append([
                '# const background markers', self.nconst_background_markers
            ])
        else:
            t.append(['Background data', 'provided via Kinship matrix'])

        t.append(['Kinship diagonal mean', self.kinship_diagonal_mean])

        if self.covariates_user_provided:
            t.append(['Covariates', 'provided by user'])
        else:
            t.append(['Covariates', 'a single column of ones'])

        t.append(['Kinship rank', self.kinship_rank])
        t.append(['# candidate markers', self.candidate_nmarkers])
        return tabulate(t, tablefmt='plain')


def genetic_preprocess(X, G, K, covariates, background):
    logger = logging.getLogger(__name__)
    logger.info("Number of candidate markers to scan: %d", X.shape[1])

    if K is not None:
        background.provided_via_variants = False
        logger.info('Covariace matrix normalization.')
        K = ascontiguousarray(K, dtype=float)
        gower_normalization(K, out=K)


    if G is not None:
        background.provided_via_variants = True
        G = ascontiguousarray(G, dtype=float)
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


def normal_scan(y, X, G=None, K=None, covariates=None, progress=True):
    """Association between genetic markers and phenotype for continuous traits.

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
    logger.info('Normal association scan has started.')
    y = ascontiguousarray(y, dtype=float)

    ii = InputInfo()
    ii.phenotype_info = NormalPhenotype(y)

    genetic_preprocess(X, G, K, covariates, ii)

    from ._canonical import CanonicalLRTScan
    lrt = CanonicalLRTScan(
        y, ii.Q0, ii.Q1, ii.S0, covariates=covariates, progress=progress)
    lrt.candidate_markers = ii.effective_X
    lrt.pvalues()
    return lrt


def binomial_scan(nsuccesses,
                  ntrials,
                  X,
                  G=None,
                  K=None,
                  covariates=None,
                  progress=True):
    """Association between genetic markers and phenotype for continuous traits.

    Matrix `X` shall contain the genetic markers (e.g., number of minor
    alleles) with rows and columns representing samples and genetic markers,
    respectively.

    The user must specify only one of the parameters `G` and `K` for defining
    the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates,
    :math:`P_c` the number of genetic markers to be tested, and :math:`P_b`
    the number of genetic markers used for Kinship estimation.

    Args:
        nsuccesses (array_like): Phenotype described by the number of
                                 successes, as non-negative integers.
                                 Dimension (:math:`N\\times 0`).
        ntrials    (array_like): Phenotype described by the number of
                                 trials, as positive integers. Dimension
                                 (:math:`N\\times 0`).
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
        progress         (bool): Shows progress. Defaults to `True`.

    Returns:
        A :class:`lim.genetics.qtl.LikelihoodRatioTest` instance.
    """

    logger = logging.getLogger(__name__)
    logger.info('Binomial association scan has started.')
    nsuccesses = ascontiguousarray(nsuccesses, dtype=float)
    ntrials = ascontiguousarray(ntrials, dtype=float)

    phenotype = BinomialPhenotype(nsuccesses, ntrials)
    background = Background()

    genetic_preprocess(X, G, K, covariates, background)
    # from .lrt import BinomialLRT
    import pdb; pdb.set_trace()
    lrt = BinomialLRT(
        nsuccesses,
        ntrials,
        ii.Q0,
        ii.Q1,
        ii.S0,
        covariates=covariates,
        progress=progress)
    lrt.candidate_markers = ii.effective_X
    lrt.pvalues()
    return lrt
