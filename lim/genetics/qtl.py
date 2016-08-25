"""Quantitative trait locus discovery."""

from __future__ import absolute_import
import sys
from tabulate import tabulate
import logging
from numpy import asarray
from numpy import asarray
from numpy import newaxis
from numpy import hstack
from numpy import sqrt
from numpy import ones
from numpy import nan
from limix_math.linalg import qs_decomposition
from limix_math.linalg import _QS_from_K_split
from ..tool.kinship import gower_normalization
from ..tool.normalize import stdnorm
from ..util.block import Block
from ..util.greek import get_greek
from .fastlmm import FastLMM

from progressbar import ProgressBar
from progressbar import Percentage

import scipy.stats as st

__all__ = ['normal_scan']


def _get_offset_covariate(covariates, n):
    return ones((n, 1)) if covariates is None else covariates


class LRT(object):

    def __init__(self, y, Q0, Q1, S0, covariates=None):

        self._logger = logging.getLogger(__name__)

        self._y = y
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0
        self._covariates = _get_offset_covariate(covariates, y.shape[0])
        self._null_model_ready = False
        self._alt_model_ready = False

        self._pvals = None
        self._lrs = None
        self._flmm = None
        self._betas = None
        self._lml_null = nan
        self._X = None
        self._lml_alt = None

    @property
    def candidate_markers(self):
        return self._X

    @candidate_markers.setter
    def candidate_markers(self, X):
        if self._X is None:
            self._X = X
            self._alt_model_ready = False
        elif np.any(self._X != X):
            self._X = X
            self._alt_model_ready = False

    def _compute_statistics(self):
        self._logger.info('Statistics computation has started.')

        self._compute_null_model()
        self._compute_alt_models()

    def _compute_alt_models(self):
        if self._alt_model_ready:
            return

        msg = "Alternative model learning "
        self._logger.info('Alternative model computation has started.')

        X = self._X
        covariates = self._covariates

        self._betas = []
        lml_alt = []
        pbar = ProgressBar(widgets=[msg, Percentage()], maxval=X.shape[1]).start()
        for i in pbar((i for i in range(X.shape[1]))):
            flmm = self._flmm.copy()
            flmm.covariates = hstack((covariates, X[:, i, newaxis]))
            flmm.learn()
            lml_alt.append(flmm.lml())
            self._betas.append(flmm.beta[-1])
        lml_alt = asarray(lml_alt)

        lml_null = self._lml_null
        lrs = -2 * lml_null + 2 * lml_alt
        chi2 = st.chi2(df=1)

        self._pvals = chi2.sf(lrs)
        self._lrs = lrs
        self._alt_model_ready = True
        self._betas = asarray(self._betas)
        self._lml_alt = lml_alt

    def _compute_null_model(self):
        if self._null_model_ready:
            return

        sys.stdout.write("Null model learning")
        self._logger.info('Null model computation has started.')

        y = self._y
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0
        covariates = self._covariates

        flmm = FastLMM(y, covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn(verbose=True)

        self._lml_null = flmm.lml()
        self._flmm = flmm

        self._null_model_ready = True

    def lml_alt(self):
        return self._lml_alt

    @property
    def effsizes(self):
        return self._betas

    def lrs(self):
        self._compute_statistics()
        return self._lrs

    def pvals(self):
        self._compute_statistics()
        return self._pvals


class PhenotypeInfo(object):

    def __init__(self, likelihood, phenotype):
        assert likelihood in ['normal', 'bernoulli', 'binomial']
        self.likelihood = likelihood
        self.phenotype = phenotype

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


def normal_scan(y, X, G=None, K=None, covariates=None, verbose=False):
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
        y          (numpy.ndarray): Phenotype. Dimension (:math:`N\\times 0`).
        X          (numpy.ndarray): Candidate genetic markers (or any other
                                    type of explanatory variable) whose
                                    association with the phenotype will be
                                    tested. Dimension (:math:`N\\times P_c`).
        G          (numpy.ndarray): Genetic markers matrix used internally for
                                    kinship estimation. Dimension
                                    (:math:`N\\times P_b`).
        K          (numpy.ndarray): Kinship matrix. Dimension
                                    (:math:`N\\times N`).
        covariates (numpy.ndarray): Covariates. Default is an offset.
                                    Dimension (:math:`N\\times S`).
    Returns:
        tuple: The estimated p-values and additional information, respectively.
    """

    logger = logging.getLogger(__name__)
    logger.info('Association scan has started.')
    y = asarray(y, dtype=float)

    ii = InputInfo()
    ii.phenotype_info = PhenotypeInfo('normal', y)

    ii.candidate_nmarkers = X.shape[1]
    ii.nconst_markers = sum(X.std(0) == 0)

    info = dict()

    ii.covariates_user_provided = covariates is not None

    if K is not None:
        logger.info('Covariace matrix normalization.')
        K = asarray(K, dtype=float)
        K = gower_normalization(K)
        info['K'] = K

    ii.background_markers_user_provided = G is not None
    if G is not None:
        logger.info('Genetic markers normalization.')
        G = asarray(G, dtype=float)
        ii.nconst_background_markers = sum(G.std(0) == 0)
        G = stdnorm(G)
        G /= sqrt(G.shape[1])
        ii.background_nmarkers = G.shape[1]
        info['G'] = G

    if G is None and K is None:
        raise Exception('G, K, and QS cannot be all None.')

    logger.info('Computing the economic eigen decomposition.')
    if K is None:
        QS = qs_decomposition(G)
    else:
        QS = _QS_from_K_split(K)

    ii.kinship_rank = len(QS[1][0])

    logger.info('Genetic marker candidates normalization.')
    X = stdnorm(X)
    info['X'] = X

    Q0, Q1 = QS[0]
    S0 = QS[1][0]

    ii.kinship_diagonal_mean = S0.sum() / Q0.shape[0]

    lrt = LRT(y, Q0, Q1, S0, covariates=covariates)
    lrt.candidate_markers = X
    info['lrs'] = lrt.lrs()
    info['effsizes'] = lrt.effsizes
    info['null_model'] = lrt._flmm
    info['lml_alt'] = lrt.lml_alt()
    return_ = (lrt.pvals(), info)

    with Block('INPUT INFO') as print_:
        print_(ii)

    with Block('NULL MODEL') as print_:
        print_(lrt._flmm)

    with Block('ALTERNATIVE MODEL') as print_:
        am = """
Phenotype:
  y_i = o_i + x_j {beta}_j + u_i + e_i

Definitions:
  {beta}_j: fixed-effect sizes of the j-th marker candidate
""".format(beta=get_greek('beta'))
        print_(am.strip())

    if verbose:
        table = [info['effsizes'], info['lml_alt'], lrt.pvals()]
        table = [list(i) for i in table]
        table = map(list, zip(*table))
        print(tabulate(table, headers=('effect-sizes', 'lml', 'pvals'), floatfmt='e'))

    return return_
