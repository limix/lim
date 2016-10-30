from __future__ import division
import logging
from numpy import ascontiguousarray

from numpy import copy
from numpy import sqrt
from numpy import ones

from ...inference import BernoulliEP
from ...inference import BinomialEP
from ...inference import PoissonEP
from ...tool.normalize import stdnorm
from ...tool.kinship import gower_normalization

from limix_math import (economic_qs, economic_qs_linear)


def bernoulli_estimate(outcomes, G=None, K=None, covariate=None):
    """Estimate the narrow-sense heritability for Bernoulli traits.

    The user must specifiy only one of the parameters G and K for defining the
    genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates, and
    :math:`P_b` the number of genetic markers used for Kinship estimation.

    :param numpy.ndarray outcomes: Phenotype. The domain has be the
                non-negative integers. Dimension (:math:`N\\times 0`).
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
    logger.info('Heritability estimation for Bernoulli traits has started.')
    outcomes = ascontiguousarray(outcomes, dtype=float)

    G, K = _background_standardize(G, K)

    if G is None and K is None:
        raise Exception('G and K cannot be all None.')

    Q0, Q1, S0 = _background_decomposition(G, K)

    if covariate is None:
        logger.debug('Inserting offset covariate.')
        covariate = ones((outcomes.shape[0], 1))

    logger.debug('Constructing EP.')
    ep = BernoulliEP(outcomes, covariate, Q0, Q1, S0)

    logger.debug('EP optimization.')
    ep.optimize()

    h2 = ep.heritability
    logger.info('Found heritability before correction: %.5f.', h2)

    return h2


def binomial_estimate(nsuccesses, ntrials, G=None, K=None, covariate=None):
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
    nsuccesses = ascontiguousarray(nsuccesses, dtype=float)
    ntrials = ascontiguousarray(ntrials, dtype=float)

    G, K = _background_standardize(G, K)

    if G is None and K is None:
        raise Exception('G and K cannot be all None.')

    Q0, Q1, S0 = _background_decomposition(G, K)

    if covariate is None:
        logger.debug('Inserting offset covariate.')
        covariate = ones((nsuccesses.shape[0], 1))

    logger.debug('Constructing EP.')
    ep = BinomialEP(nsuccesses, ntrials, covariate, Q0, Q1, S0)

    logger.debug('EP optimization.')
    ep.optimize()

    h2 = ep.heritability
    logger.info('Found heritability before correction: %.5f.', h2)

    return h2


def poisson_estimate(nsuccesses, G=None, K=None, covariate=None):
    """Estimate the so-called narrow-sense heritability.

    It supports Bernoulli and Poisson phenotypes (see `outcome_type`).
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
    nsuccesses = ascontiguousarray(nsuccesses, dtype=float)

    G, K = _background_standardize(G, K)

    if G is None and K is None:
        raise Exception('G and K cannot be all None.')

    Q0, Q1, S0 = _background_decomposition(G, K)

    if covariate is None:
        logger.debug('Inserting offset covariate.')
        covariate = ones((nsuccesses.shape[0], 1))

    logger.debug('Constructing EP.')
    ep = PoissonEP(nsuccesses, covariate, Q0, Q1, S0)

    logger.debug('EP optimization.')
    ep.optimize()

    h2 = ep.heritability
    logger.info('Found heritability before correction: %.5f.', h2)

    return h2


def _background_standardize(G, K):
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
    if G:
        (Q, S0) = economic_qs_linear(G)
    else:
        (Q, S0) = economic_qs(K)

    Q0 = Q[0]
    Q1 = Q[1]
    S0 /= S0.mean()

    return Q0, Q1, S0
