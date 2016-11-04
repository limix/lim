from __future__ import absolute_import
from __future__ import unicode_literals

from cachetools import LRUCache
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey

from numpy import empty

from .qtl import QTLScan
from ...inference import FastLMM


class NormalQTLScan(QTLScan):
    def __init__(self, y, covariates, X, Q0, Q1, S0):
        super(NormalQTLScan, self).__init__(X)
        self._cache_compute_null_model = LRUCache(maxsize=1)
        self._cache_compute_alt_models = LRUCache(maxsize=1)
        self._y = y
        self._covariates = covariates
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0

        self._flmm = None

    @cachedmethod(
        attrgetter('_cache_compute_null_model'),
        key=lambda self, progress: hashkey(self))
    def _compute_null_model(self, progress):
        y = self._y
        covariates = self._covariates
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0

        flmm = FastLMM(y, covariates, QS=((Q0, Q1), S0))
        flmm.learn(progress=progress)
        self._flmm = flmm
        self._null_lml = flmm.lml()

    @cachedmethod(
        attrgetter('_cache_compute_alt_models'),
        key=lambda self, progress: hashkey(self))
    def _compute_alt_models(self, progress):
        n, p = self._X.shape
        nc = self._covariates.shape[1]
        self._alt_lmls = empty(p)
        self._effect_sizes = empty(p)
        M = empty((n, nc + 1))
        M[:, :nc] = self._covariates
        for i in range(p):
            M[:, nc] = self._X[:, i]
            flmm = self._flmm.copy()
            flmm.covariates = M
            flmm.learn()
            self._alt_lmls[i] = flmm.lml()
            self._effect_sizes[i] = flmm.beta[-1]
