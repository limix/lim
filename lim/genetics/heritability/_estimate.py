from __future__ import division
import logging
from numpy import ascontiguousarray

from numpy import copy
from numpy import sqrt
from numpy import ones

from numpy_sugar.linalg import (economic_qs, economic_qs_linear)

def estimate(phenotype, G=None, K=None, covariates=None):
    """Estimate the so-called narrow-sense heritability.

    It supports Bernoulli and Binomial phenotypes (see `outcome_type`).
    The user must specifiy only one of the parameters G, K, and QS for
    defining the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates, and
    :math:`P_b` the number of genetic markers used for Kinship estimation.

    :param numpy.ndarray y: Phenotype. The domain has be the non-negative
                          integers. Dimension (:math:`N\\times 0`).
    :param numpy.ndarray G: Genetic markers matrix used internally for kinship
                    estimation. Dimension (:math:`N\\times P_b`).
    :param numpy.ndarray K: Kinship matrix. Dimension (:math:`N\\times N`).
    :param tuple QS: Economic eigen decomposition of the Kinship matrix.
    :param numpy.ndarray covariate: Covariates. Default is an offset.
                                  Dimension (:math:`N\\times S`).
    :param object oucome_type: Either :class:`limix_qep.Bernoulli` (default)
                               or a :class:`limix_qep.Binomial` instance.
    :param float prevalence: Population rate of cases for dichotomous
                             phenotypes. Typically useful for case-control
                             studies.
    :return: a tuple containing the estimated heritability and additional
             information, respectively.
    """
    logger = logging.getLogger(__name__)
    logger.info('Heritability estimation has started.')

    G, K = _background_standardize(G, K)

    if G is None and K is None:
        raise Exception('G and K cannot be all None.')

    Q0, Q1, S0 = _background_decomposition(G, K)

    if covariates is None:
        logger.debug('Inserting offset covariate.')
        covariates = ones((phenotype.sample_size, 1))

    logger.debug('Constructing EP.')
    from limix_inference.glmm import ExpFamEP
    ep = ExpFamEP(phenotype.to_likelihood(), covariates, Q0, Q1, S0)

    logger.debug('EP optimization.')
    ep.optimize()

    h2 = ep.heritability
    logger.info('Found heritability before correction: %.5f.', h2)

    return h2

def _background_standardize(G, K):
    from ...tool.normalize import stdnorm
    from ...tool.kinship import gower_normalization
    logger = logging.getLogger(__name__)

    if K is not None:
        logger.debug('Covariace matrix normalization.')
        K = copy(K, 'C')
        K = ascontiguousarray(K, dtype=float)
        gower_normalization(K, K)

    if G is not None:
        logger.debug('Genetic markers normalization.')
        G = copy(G, 'C')
        G = ascontiguousarray(G, dtype=float)
        stdnorm(G, 0, out=G)
        G /= sqrt(G.shape[1])

    return (G, K)


def _background_decomposition(G, K):
    if G is None:
        (Q, S0) = economic_qs(K)
    else:
        (Q, S0) = economic_qs_linear(G)

    Q0 = Q[0]
    Q1 = Q[1]
    S0 /= S0.mean()

    return Q0, Q1, S0
