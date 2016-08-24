from __future__ import absolute_import
from tabulate import tabulate
import logging
from numpy import asarray
from numpy import asarray
from numpy import sqrt
from numpy import ones
from numpy import nan
from limix_math.linalg import qs_decomposition
from limix_math.linalg import _QS_from_K_split
from ..tool.kinship import gower_normalization
from .fastlmm import FastLMM

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
        self._ep = None
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

        import ipdb; ipdb.set_trace()
        self._logger.info('Alternative model computation has started.')

        X = self._X
        covariates = self._covariates

        lml_alt = []
        for x in X:
            flmm = self._flmm.copy()
            flmm.covariates = x
            flmm.learn()
            lml_alt.append(flmm.lml())
        lml_alt = asarray(lml_alt)

        lml_null = self._lml_null
        lrs = -2 * lml_null + 2 * lml_alt
        chi2 = st.chi2(df=1)

        self._pvals = chi2.sf(lrs)
        self._lrs = lrs
        self._alt_model_ready = True

    def _compute_null_model(self):
        if self._null_model_ready:
            return

        self._logger.info('Null model computation has started.')

        y = self._y
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0
        covariates = self._covariates

        flmm = FastLMM(y, covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn()

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
        if self._null_model_only:
            return []
        return self._lrs

    def pvals(self):
        self._compute_statistics()
        if self._null_model_only:
            return []
        return self._pvals


def normal_scan(y, X, G=None, K=None, covariates=None):
    """Perform association scan between genetic markers and phenotype.

    Matrix `X` shall contain the genetic markers (e.g., number of minor alleles)
    with rows and columns representing samples and genetic markers,
    respectively.

    The user must specify only one of the parameters `G` and `K` for defining
    the genetic background.

    Let :math:`N` be the sample size, :math:`S` the number of covariates,
    :math:`P_c` the number of genetic markers to be tested, and :math:`P_b`
    the number of genetic markers used for Kinship estimation.

    Args:
        y (numpy.ndarray): Phenotype. Dimension (:math:`N\\times 0`).
        X          (numpy.ndarray): Candidate genetic markers (or any other
                                    type of explanatory variable) whose
                                    association with the phenotype will be tested. Dimension
                                    (:math:`N\\times P_c`).
        G          (numpy.ndarray): Genetic markers matrix used internally for kinship
                                    estimation. Dimension (:math:`N\\times P_b`).
        K          (numpy.ndarray): Kinship matrix. Dimension (:math:`N\\times N`).
        covariates  (numpy.ndarray): Covariates. Default is an offset.
                                    Dimension (:math:`N\\times S`).
    Returns:
        tuple: The estimated p-values and additional information, respectively.
    """

    logger = logging.getLogger(__name__)
    logger.info('Association scan has started.')
    y = asarray(y, dtype=float)

    print("Number of candidate markers to scan: %d" % X.shape[1])

    info = dict()

    if K is not None:
        logger.info('Covariace matrix normalization.')
        K = asarray(K, dtype=float)
        K = gower_normalization(K)
        info['K'] = K

    if G is not None:
        logger.info('Genetic markers normalization.')
        G = asarray(G, dtype=float)
        G = G - G.mean(0)
        s = G.std(0)
        ok = s > 0.
        G[:, ok] /= s[ok]
        G /= sqrt(G.shape[1])
        info['G'] = G

    if G is None and K is None:
        raise Exception('G, K, and QS cannot be all None.')

    logger.info('Computing the economic eigen decomposition.')
    if K is None:
        QS = qs_decomposition(G)
    else:
        QS = _QS_from_K_split(K)

    logger.info('Genetic marker candidates normalization.')
    X = X - X.mean(0)
    s = X.std(0)
    ok = s > 0.
    X[:, ok] /= s[ok]
    info['X'] = X

    Q0, Q1 = QS[0]
    S0 = QS[1][0]

    print("Scan has began...")
    lrt = LRT(y, Q0, Q1, S0, covariates=covariates)
    lrt.candidate_markers = X
    info['lrs'] = lrt.lrs()
    # info['effsizes'] = lrt.effsizes
    # info['ep_null_model'] = lrt._ep
    # info['lml_alt'] = lrt.lml_alt()
    # return_ = (lrt.pvals(), info)
    # print("Scan has finished.")
    #
    # print("-------------------------- NULL MODEL --------------------------")
    # print(lrt._ep)
    # print("----------------------------------------------------------------")
    # print("")
    #
    # table = [info['effsizes'], info['lml_alt'], info['lrs'], lrt.pvals()]
    # table = [list(i) for i in table]
    # table = map(list, zip(*table))
    # print("---------------------- ALTERNATIVE MODELs ----------------------")
    # print(tabulate(table, headers=('EffSiz', 'LML', 'LR', 'Pval')))
    # print("----------------------------------------------------------------")
    #
    # return return_
