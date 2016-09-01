"""Quantitative trait locus discovery."""

from __future__ import absolute_import

import sys

import logging

from tabulate import tabulate

from progressbar import ProgressBar
from progressbar import Percentage

from numpy import asarray
from numpy import asarray
from numpy import newaxis
from numpy import hstack
from numpy import sqrt
from numpy import ones
from numpy import nan

from limix_math.linalg import qs_decomposition
from limix_math.linalg import _QS_from_K_split

from ...tool.kinship import gower_normalization
from ...tool.normalize import stdnorm
from ...util.block import Block
from ...util.greek import get_greek
from ..core import FastLMM
from .lrt import NormalLRT
from .lrt import BinomialLRT


class NormalPhenotypeInfo(object):

    def __init__(self, y):
        self.y = y

    def get_info(self):
        lik = self.likelihood
        t = [['Likelihood', lik[0].upper() + lik[1:]]]
        if self.likelihood == 'normal':
            t.append(['Phenotype mean', self.phenotype.mean()])
            t.append(['Phenotype std', self.phenotype.std()])
            mima = (self.phenotype.min(), self.phenotype.max())
            t.append(['Phenotype (min, max)', mima])
        return t


class BinomialPhenotypeInfo(object):

    def __init__(self, nsuccesses, ntrials):
        self.nsuccesses = nsuccesses
        self.ntrials = ntrials

    def get_info(self):
        lik = self.likelihood
        t = [['Likelihood', lik[0].upper() + lik[1:]]]
        if self.likelihood == 'normal':
            t.append(['Phenotype mean', self.phenotype.mean()])
            t.append(['Phenotype std', self.phenotype.std()])
            mima = (self.phenotype.min(), self.phenotype.max())
            t.append(['Phenotype (min, max)', mima])
        return t


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
            t.append(['# const background markers',
                      self.nconst_background_markers])
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


def genetic_preprocess(X, G, K, covariates, input_info):
    logger = logging.getLogger(__name__)
    logger.info("Number of candidate markers to scan: %d", X.shape[1])

    input_info.candidate_nmarkers = X.shape[1]
    input_info.nconst_markers = sum(X.std(0) == 0)

    input_info.covariates_user_provided = covariates is not None

    if K is not None:
        logger.info('Covariace matrix normalization.')
        K = asarray(K, dtype=float)
        K = gower_normalization(K)

    input_info.background_markers_user_provided = G is not None
    if G is not None:
        logger.info('Genetic markers normalization.')
        G = asarray(G, dtype=float)
        input_info.nconst_background_markers = sum(G.std(0) == 0)
        G = stdnorm(G)
        G /= sqrt(G.shape[1])
        input_info.background_nmarkers = G.shape[1]

    if G is None and K is None:
        raise Exception('G, K, and QS cannot be all None.')

    logger.info('Computing the economic eigen decomposition.')
    if K is None:
        QS = qs_decomposition(G)
    else:
        QS = _QS_from_K_split(K)

    input_info.Q = QS[0]
    input_info.S = QS[1]

    input_info.kinship_rank = len(QS[1][0])

    Q0, Q1 = QS[0]
    S0 = QS[1][0]

    input_info.kinship_diagonal_mean = S0.sum() / Q0.shape[0]

    logger.info('Genetic marker candidates normalization.')
    X = stdnorm(X)

    input_info.effective_X = X
    input_info.effective_G = G
    input_info.effective_K = K


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
        A :class:`lim.genetics.qtl.LikelihoodRatioTest` instance.
    """
    logger = logging.getLogger(__name__)
    logger.info('Normal association scan has started.')
    y = asarray(y, dtype=float)

    ii = InputInfo()
    ii.phenotype_info = NormalPhenotypeInfo(y)

    genetic_preprocess(X, G, K, covariates, ii)

    lrt = NormalLRT(y, ii.Q[0], ii.Q[1], ii.S[0], covariates=covariates,
                    progress=progress)
    lrt.candidate_markers = ii.effective_X
    lrt.pvals()
    return lrt


def binomial_scan(nsuccesses, ntrials, X, G=None, K=None, covariates=None,
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
    nsuccesses = asarray(nsuccesses, dtype=float)
    ntrials = asarray(ntrials, dtype=float)

    ii = InputInfo()
    ii.phenotype_info = BinomialPhenotypeInfo(nsuccesses, ntrials)

    genetic_preprocess(X, G, K, covariates, ii)

    lrt = BinomialLRT(nsuccesses, ntrials, ii.Q[0], ii.Q[1], ii.S[0],
                      covariates=covariates, progress=progress)
    lrt.candidate_markers = ii.effective_X
    lrt.pvals()
    return lrt
