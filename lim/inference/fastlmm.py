from __future__ import division

from numpy import exp
from numpy import clip
from numpy import isfinite
from numpy import atleast_2d
from numpy import all as all_
from numpy import set_printoptions

from limix_math import economic_qs_linear

from optimix import maximize_scalar
from optimix import Function
from optimix import Scalar

from ..util.transformation import DesignMatrixTrans
from .fastlmm_core import FastLMMCore
from ..genetics.model import CanonicalModel


class FastLMM(Function):
    def __init__(self, y, covariates, X=None, QS=None):
        super(FastLMM, self).__init__(logistic=Scalar(0.0))

        assert (X is None) != (QS is None)
        if not all_(isfinite(y)):
            raise ValueError("There are non-finite values in the phenotype.")

        self._trans = None
        if QS is None:
            self._trans = DesignMatrixTrans(X)
            X = self._trans.transform(X)
            QS = economic_qs_linear(X)
            self._X = X

        self._flmmc = FastLMMCore(y, covariates, QS[0][0], QS[0][1], QS[1])
        self.set_nodata()

    @property
    def M(self):
        return self._flmmc.M

    @M.setter
    def M(self, v):
        self._flmmc.M = v

    def copy(self):
        o = FastLMM.__new__(FastLMM)
        super(FastLMM, o).__init__(logistic=Scalar(self.get('logistic')))
        o._flmmc = self._flmmc.copy()
        o._trans = self._trans
        o.set_nodata()
        return o

    def _delta(self):
        v = clip(self.get('logistic'), -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, 1e-5, 1 - 1e-5)

    @property
    def heritability(self):
        t = (self.fixed_effects_variance + self.genetic_variance +
             self.environmental_variance)
        return self.genetic_variance / t

    @property
    def fixed_effects_variance(self):
        return self._flmmc.m.var()

    @property
    def genetic_variance(self):
        return self._flmmc.scale * (1 - self._flmmc.delta)

    @property
    def environmental_variance(self):
        return self._flmmc.scale * self._flmmc.delta

    @property
    def beta(self):
        return self._flmmc.beta

    @property
    def m(self):
        return self._flmmc.m

    def learn(self, progress=None):
        maximize_scalar(self, progress=progress)
        self._flmmc.delta = self._delta()

    def value(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()

    def lml(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()

    def predict(self, covariates, Xp):
        covariates = atleast_2d(covariates)
        Xp = atleast_2d(Xp)
        Xp = self._trans.transform(Xp)
        Cp = Xp.dot(self._X.T)
        Cpp = Xp.dot(Xp.T)
        return self._flmmc.predict(covariates, Cp, Cpp)

    def model(self):
        total_var = (self.fixed_effects_variance + self.genetic_variance +
                     self.environmental_variance)
        return CanonicalModel(self.beta, self.fixed_effects_variance,
                              self.heritability, self.genetic_variance,
                              self.environmental_variance, total_var)
