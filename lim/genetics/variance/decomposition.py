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
    # ii.phenotype_info = NormalPhenotypeInfo(y)

    GK = normalize_covariance_list(GK)
    preprocess(GK, covariates, ii)


    vd = NormalVarDec(y, input_info.effective_GK, covariates=covariates,
                      progress=progress)

    # genetic_preprocess(X, G, K, covariates, ii)
    #
    # lrt = NormalLRT(y, ii.Q[0], ii.Q[1], ii.S[0], covariates=covariates,
    #                 progress=progress)
    # lrt.candidate_markers = ii.effective_X
    # lrt.pvals()
    # return lrt
    return vd

class VarDec(object):

    def __init__(self, GK, covariates=None, progress=True):

        self._logger = logging.getLogger(__name__)

        self._progress = progress
        self._X = None
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0
        self._covariates = _offset_covariate(covariates, Q0.shape[0])
        self._candidate_effect_sizes = None

        self._null_lml = None
        self._alt_lmls = None

        self._null_model_ready = False
        self._alt_model_ready = False

    @property
    def candidate_markers(self):
        """Candidate markers.

        :getter: Returns candidate markers
        :setter: Sets candidate markers
        :type: `array_like` (:math:`N\\times P_c`)
        """

        return self._X

    @candidate_markers.setter
    def candidate_markers(self, X):
        self._X = X
        self._alt_model_ready = False

    def _compute_statistics(self):
        self._logger.info('Likelihood-ratio test Statistics computation: ' +
                          'has started.')
        self._compute_null_model()
        self._compute_alt_models()

    def _compute_null_model(self):
        if self._null_model_ready:
            return
        self._logger.info('Null model computation has started.')

        if self._progress:
            print("Null model fitting: ")
            progress = ProgressBar(widgets=["  ", Counter(),
                                            " function evaluations"],
                                   max_value=UnknownLength)
        else:
            progress = NullBar()

        self._learn_null_model(progress)

        self._null_model_ready = True

    def _compute_alt_models(self):
        if self._alt_model_ready:
            return
        self._logger.info('Alternative model computation has started.')

        nmarkers = self._X.shape[1]

        if self._progress:
            print("Candidate markers analysis:")
            progress = ProgressBar(widgets=["  ", AdaptiveETA()],
                                   max_value=nmarkers)
        else:
            progress = NullBar()

        self._prepare_for_scan()
        for i in progress((i for i in range(nmarkers))):
            self._process_marker(i)

        self._alt_model_ready = True

    def null_lml(self):
        """Log marginal likelihood for the null hypothesis."""
        self._compute_statistics()
        return self._null_lml

    def alt_lmls(self):
        """Log marginal likelihoods for the alternative hypothesis."""
        self._compute_statistics()
        return self._alt_lmls

    def candidate_effect_sizes(self):
        """Effect size for candidate markers."""
        self._compute_statistics()
        return self._candidate_effect_sizes

    def pvals(self):
        """Association p-value for candidate markers."""
        self._compute_statistics()

        lml_alts = self.alt_lmls()
        lml_null = self.null_lml()

        lrs = -2 * lml_null + 2 * asarray(lml_alts)

        from scipy.stats import chi2
        chi2 = chi2(df=1)

        return chi2.sf(lrs)

    def null_model(self):
        """Model of the null hypothesis."""
        raise NotImplementError

    def alt_model(self):
        """Model of the alternative hypotheses."""
        raise NotImplementError

    def __str__(self):
        snull = str(self.null_model())
        snull = 'Null model:\n\n' + _indent(snull)

        salt = self.alt_model()
        salt = 'Alternative model:\n\n' + _indent(salt)

        sces = 'Candidate effect sizes:\n'
        sces += _indent(quantile_summary(self._candidate_effect_sizes))
        sces = _indent(sces)

        salmls = 'Candidate log marginal likelihoods:\n'
        salmls += _indent(quantile_summary(self._alt_lmls))
        salmls = _indent(salmls)

        spval = 'Candidate p-values:\n'
        spval += _indent(quantile_summary(self.pvals(), "e"))
        spval = _indent(spval)

        return '\n\n'.join([snull, salt, sces, salmls, spval])

class NormalVarDec(VarDec):

    def __init__(self, y, Q0, Q1, S0, covariates=None, progress=True):
        super(NormalVarDec, self).__init__(Q0, Q1, S0, covariates=covariates,
                                        progress=progress)
        self._y = y

    def _learn_null_model(self, progress):
        y = self._y
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0
        covariates = self._covariates

        flmm = FastLMM(y, covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn(progress=progress)
        self._flmm = flmm
        self._null_lml = flmm.lml()

    def _prepare_for_scan(self):
        self._alt_lmls = []
        self._candidate_effect_sizes = []

    def _process_marker(self, i):
        x = self._X[:, i]
        flmm = self._flmm.copy()
        flmm.covariates = hstack((self._covariates, x[:, newaxis]))
        flmm.learn()
        self._alt_lmls.append(flmm.lml())
        self._candidate_effect_sizes.append(flmm.beta[-1])

    def null_model(self):
        return self._flmm.model()

    def alt_model(self):
        s = "Phenotype:\n"
        s += "    y_i = o_i + b_j x_{i,j} + u_i + e_i\n\n"
        s += "Definitions:\n"
        s += "    b_j    : effect-size of the j-th candidate marker\n"
        s += "    x_{i,j}: j-th candidate marker of the i-th sample\n"
        return s
