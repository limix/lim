from __future__ import division

import logging

from numpy import ascontiguousarray, copy, ones, sqrt

from limix_inference.glmm import GLMM
from limix_inference.lmm import LMM
from numpy_sugar.linalg import economic_qs, economic_qs_linear


def estimate(y, likname, X=None, G=None, K=None):
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

    if not isinstance(y, tuple):
        y = (y, )

    if G is None and K is None:
        raise ValueError('G and K cannot be all None.')

    G, K = _background_standardize(G, K)

    QS = _background_decomposition(G, K)

    if X is None:
        X = ones((len(y[0]), 1))

    import pdb
    pdb.set_trace()
    if likname.lower() == 'normal':
        model = LMM(y, X, QS=QS)
    else:
        model = GLMM(y, likname, X, QS=(QS[0][0], QS[1]))

    model.feed().maximize()

    return model.heritability


def _background_standardize(G, K):
    from ..tool.normalize import stdnorm
    from ..tool.kinship import gower_normalization
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
        QS = economic_qs(K)
    else:
        QS = economic_qs_linear(G)

    S0 = QS[1]
    S0 /= S0.mean()

    return QS
