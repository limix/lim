from __future__ import absolute_import
from __future__ import unicode_literals

from cachetools import LRUCache
from operator import attrgetter
from cachetools import cachedmethod
from cachetools.keys import hashkey

from .qtl import QTLScan
from ...inference import BinomialEP


class BinomialQTLScan(QTLScan):
    def __init__(self, nsuccesses, ntrials, covariates, X, Q0, Q1, S0):
        super(BinomialQTLScan, self).__init__(X)
        self._cache_compute_null_model = LRUCache(maxsize=1)
        self._cache_compute_alt_models = LRUCache(maxsize=1)
        self._nsuccesses = nsuccesses
        self._ntrials = ntrials
        self._covariates = covariates
        self._Q0 = Q0
        self._Q1 = Q1
        self._S0 = S0

        self._fixed_ep = None

    @cachedmethod(
        attrgetter('_cache_compute_null_model'),
        key=lambda self, progress: hashkey(self))
    def _compute_null_model(self, progress):
        nsuccesses = self._nsuccesses
        ntrials = self._ntrials
        covariates = self._covariates
        Q0, Q1 = self._Q0, self._Q1
        S0 = self._S0

        ep = BinomialEP(nsuccesses, ntrials, covariates, Q0=Q0, Q1=Q1, S0=S0)
        ep.optimize(progress=progress)
        self._null_lml = ep.lml()
        self._fixed_ep = ep.fixed_ep()

    @cachedmethod(
        attrgetter('_cache_compute_alt_models'),
        key=lambda self, progress: hashkey(self))
    def _compute_alt_models(self, progress):
        fep = self._fixed_ep
        covariates = self._covariates
        X = self._X
        self._alt_lmls, self._effect_sizes = fep.compute(covariates, X)
