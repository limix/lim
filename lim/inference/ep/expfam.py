from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import clip, full
from numpy.linalg import lstsq

from ...inference import FastLMM
from .liknorm import create_liknorm
from .ep import EP

class ExpFamEP(EP):
    def __init__(self, phenotype, covariates, Q0, Q1, S0):
        super(ExpFamEP, self).__init__(covariates, Q0, S0, True)
        self._logger = logging.getLogger(__name__)

        self._Q1 = Q1
        self._moments = create_liknorm(phenotype.likelihood_name, 350)

        h2, m = _initialize(phenotype, covariates, Q0, Q1, S0)

        n = phenotype.sample_size

        self._tbeta = lstsq(self._tM, full(n, m))[0]
        self.delta = 1 - h2
        self.v = 1.

    def _tilted_params(self):
        nsuccesses = self._nsuccesses
        ntrials = self._ntrials
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz

        self._moments.binomial(nsuccesses, ntrials, ceta, ctau, lmom0,
                               self._hmu, self._hvar)


def _initialize(phenotype, covariates, Q0, Q1, S0):
    y = phenotype.to_normal()
    flmm = FastLMM(y, covariates, Q0, Q1, S0)
    flmm.learn()
    gv = flmm.genetic_variance
    nv = flmm.environmental_variance
    h2 = gv / (gv + nv)
    return clip(h2, 0.01, 0.9), flmm.m
