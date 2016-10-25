from __future__ import absolute_import

from collections import OrderedDict

import logging

from tabulate import tabulate

from progressbar import ProgressBar
from progressbar import NullBar
from progressbar import Percentage
from progressbar import UnknownLength
from progressbar import Counter
from progressbar import AdaptiveETA


from numpy import asarray
from numpy import sqrt
from numpy import ones
from numpy.linalg import cholesky


# from ..core import SlowLMM
from limix_math.linalg import economic_svd

from ...tool.kinship import gower_normalization
from ...tool.normalize import stdnorm
from ...inference import SlowLMM
from ...mean import LinearMean
from ...cov import LinearCov
from ...cov import SumCov


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


def tuple_it(GK):
    r = []
    for GKi in GK:
        if isinstance(GKi, tuple):
            r.append(GKi)
        else:
            r.append((GKi, False))
    return tuple(r)


def normalize_covariance_list(GK):
    if not isinstance(GK, (list, tuple, dict)):
        GK = (GK,)

    GK = tuple_it(GK)

    if not isinstance(GK, dict):
        GK = [('K%d' % i, GK[i]) for i in range(len(GK))]
        GK = OrderedDict(GK)

    return GK


def preprocess(GK, covariates, input_info):
    logger = logging.getLogger(__name__)

    input_info.covariates_user_provided = covariates is not None

    for (name, GKi) in iter(GK.items()):
        if GKi[1]:
            logger.info('Covariace matrix normalization on %s.' % name)
            K = asarray(GKi[0], dtype=float)
            K = gower_normalization(K)
            GK[name] = (K, True)
        else:
            logger.info('Genetic markers normalization on %s.' % name)
            G = asarray(GKi[0], dtype=float)
            G = stdnorm(G)
            G /= sqrt(G.shape[1])
            GK[name] = (G, False)

    input_info.effective_GK = GK


def normal_decomposition(y, GK, covariates=None, progress=True):
    logger = logging.getLogger(__name__)
    logger.info('Normal variance decomposition scan has started.')
    y = asarray(y, dtype=float)

    ii = InputInfo()
    # ii.phenotype_info = NormalPhenotypeInfo(y)

    GK = normalize_covariance_list(GK)
    preprocess(GK, covariates, ii)


    vd = NormalVarDec(y, ii.effective_GK, covariates=covariates,
                      progress=progress)

    vd.learn()
    # genetic_preprocess(X, G, K, covariates, ii)
    #
    # lrt = NormalLRT(y, ii.Q[0], ii.Q[1], ii.S[0], covariates=covariates,
    #                 progress=progress)
    # lrt.candidate_markers = ii.effective_X
    # lrt.pvals()
    # return lrt
    return vd

def _offset_covariate(covariates, n):
    return ones((n, 1)) if covariates is None else covariates

class VarDec(object):

    def __init__(self, K, covariates=None, progress=True):
        self._logger = logging.getLogger(__name__)

        self._progress = progress
        self._X = None
        self._K = K
        n = list(K.items())[0][1][0].shape[0]
        self._covariates = _offset_covariate(covariates, n)

    def learn(self):
        self._logger.info('Variance decomposition computation: has started.')
        if self._progress:
            print("Null model fitting: ")
            progress = ProgressBar(widgets=["  ", Counter(),
                                            " function evaluations"],
                                   max_value=UnknownLength)
        else:
            progress = NullBar()

        self._learn(progress)

class NormalVarDec(VarDec):

    def __init__(self, y, K, covariates=None, progress=True):
        super(NormalVarDec, self).__init__(K, covariates=covariates,
                                           progress=progress)
        self._y = y

        mean = LinearMean(self._covariates.shape[1])
        mean.set_data(self._covariates)

        covs = []
        for Ki in iter(K.items()):
            c = LinearCov()
            if Ki[1][1]:
                G = economic_svd(Ki[1][0])
            else:
                G = Ki[1][0]

            c.set_data((G, G))
            covs.append(c)

        cov = SumCov(covs)

        self._lmm = SlowLMM(y, mean, cov)

    def _learn(self, progress):
        self._lmm.feed().maximize()
