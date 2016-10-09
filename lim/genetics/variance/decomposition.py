from __future__ import absolute_import

from collections import OrderedDict

import logging

from tabulate import tabulate

from numpy import asarray
from numpy import sqrt


# from ..core import SlowLMM

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

    GK = normalize_covariance_list(GK)
    preprocess(GK, covariates, ii)
