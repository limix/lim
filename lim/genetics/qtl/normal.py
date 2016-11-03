from __future__ import absolute_import
from __future__ import unicode_literals

from numpy import concatenate
from numpy import asarray
from numpy import newaxis
from numpy import hstack
from numpy import array
from numpy import ones

from .qtl import QTLScan
from ...inference import FastLMM

class NormalQTLScan(QTLScan):
    def __init__(self, y, X, Q0, Q1, S0, covariates=None):
        super(NormalQTLScan, self).__init__(X)
        self._y = y
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0
        self._covariates = offset_covariate(covariates, len(y))

    def _compute_null_model(self, progress):
        y = self._y
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0
        covariates = self._covariates

        flmm = FastLMM(y, covariates, QS=((Q0, Q1), (S0, )))
        flmm.learn(progress=progress)
        self._flmm = flmm
        self._null_lml = flmm.lml()

    def _compute_alt_models(self, progress):
        self._alt_lmls = []
        self._candidate_effect_sizes = []
        for i in range(self._X.shape[1]):
            x = self._X[:, i]
            flmm = self._flmm.copy()
            flmm.covariates = hstack((self._covariates, x[:, newaxis]))
            flmm.learn()
            self._alt_lmls.append(flmm.lml())
            self._candidate_effect_sizes.append(flmm.beta[-1])
