from __future__ import absolute_import

import sys

import logging

from numpy import asarray
from numpy import newaxis
from numpy import hstack
from numpy import ones

from progressbar import ProgressBar
from progressbar import NullBar
from progressbar import Percentage
from progressbar import UnknownLength
from progressbar import Counter
from progressbar import AdaptiveETA

from ..core import FastLMM

from ...util import quantile_summary


def _offset_covariate(covariates, n):
    return ones((n, 1)) if covariates is None else covariates


def _indent(msg):
    return '\n'.join(['    ' + s for s in msg.split('\n')])


class LikelihoodRatioTest(object):

    def __init__(self, Q0, Q1, S0, covariates=None, progress=True):

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


class NormalLRT(LikelihoodRatioTest):

    def __init__(self, y, Q0, Q1, S0, covariates=None, progress=True):
        super(NormalLRT, self).__init__(Q0, Q1, S0, covariates=covariates,
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


class BinomialLRT(LikelihoodRatioTest):

    def __init__(self, nsuccesses, ntrials, Q0, Q1, S0, covariates=None,
                 progress=True):
        super(BinomialLRT, self).__init__(X, Q0, Q1, S0, covariates=covariates,
                                          progress=progress)
        self._nsuccess = nsuccesses
        self._ntrials = ntrials

    def candidate_effect_sizes(self):
        raise NotImplementedError

    def _prepare_for_processing_nmarkers(self):
        pass

    def _process_marker(self, x):
        pass

    def null_model(self):
        return self._ep.model()
