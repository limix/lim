from __future__ import absolute_import, unicode_literals

import logging
from copy import copy
from operator import attrgetter

from numpy import asarray, empty, nan

from limix_inference.glmm import ExpFamEP
from limix_inference.lmm import FastLMM
from numpy_sugar.linalg import economic_qs

from ..phenotype import NormalPhenotype

class QTLScan(object):
    def __init__(self, phenotype, covariates, X, Q0, Q1, S0, fast=False):
        self._logger = logging.getLogger(__name__)
        self.progress = True

        self._valid_null_model = False
        self._valid_alt_models = False
        self._phenotype = phenotype
        self._covariates = covariates
        self._X = X
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0
        self._method = None
        self._null_lml = nan
        self._alt_lmls = None
        self._effect_sizes = None
        self._fast = fast

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
        self._valid_alt_models = False

    def compute_statistics(self):
        self._compute_null_model()
        self._compute_alt_models()

    def _compute_null_model(self):
        if self._valid_null_model:
            return

        covariates = self._covariates
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0

        method = _get_method(self._phenotype, Q0, Q1, S0, covariates)
        method.learn(progress=self.progress)

        self._method = method
        self._null_lml = method.lml(self._fast)

        self._valid_null_model = True

    def _compute_alt_models(self):
        if self._valid_alt_models:
            return

        if self._fast:
            al, es = _fast_scan(self._method, self._covariates, self._X,
                                self.progress)
        else:
            al, es = _slow_scan(self._method, self._covariates, self._X,
                                self.progress)

        self._alt_lmls, self._effect_sizes = al, es

        self._valid_alt_models = True

    def null_lml(self):
        """Log marginal likelihood for the null hypothesis."""
        self.compute_statistics()
        return self._null_lml

    def alt_lmls(self):
        """Log marginal likelihoods for the alternative hypothesis."""
        self.compute_statistics()
        return self._alt_lmls

    def candidate_effect_sizes(self):
        """Effect size for candidate markers."""
        self.compute_statistics()
        return self._effect_sizes

    def pvalues(self):
        """Association p-value for candidate markers."""
        self.compute_statistics()

        lml_alts = self.alt_lmls()
        lml_null = self.null_lml()

        lrs = -2 * lml_null + 2 * asarray(lml_alts)

        from scipy.stats import chi2
        chi2 = chi2(df=1)

        return chi2.sf(lrs)

def _get_method(phenotype, Q0, Q1, S0, covariates):

    if phenotype.likelihood_name.lower() == 'normal':
        y = phenotype.outcome
        method = FastLMM(y, Q0=Q0, Q1=Q1, S0=S0, covariates=covariates)
    else:
        y = phenotype.to_likelihood()
        overdispersion = y.name != 'Bernoulli'
        method = ExpFamEP(y, covariates, Q0=Q0, Q1=Q1, S0=S0,
                          overdispersion=overdispersion)

    return method

def _slow_scan(method, covariates, X, progress):

    n, p = X.shape
    nc = covariates.shape[1]

    alt_lmls = empty(p)
    effect_sizes = empty(p)

    M = empty((n, nc + 1))
    M[:, :nc] = covariates

    for i in range(p):
        M[:, nc] = X[:, i]
        m = method.copy()
        m.M = M
        m.learn(progress=False)
        alt_lmls[i] = m.lml()
        effect_sizes[i] = m.beta[-1]

    return alt_lmls, effect_sizes

def _fast_scan(method, covariates, X, progress):
    nlt = method.get_normal_likelihood_trick()

    alt_lmls, effect_sizes = nlt.fast_scan(X)

    return alt_lmls, effect_sizes
