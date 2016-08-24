from __future__ import division

from numpy import exp
from numpy import clip
from numpy import isfinite
from numpy import atleast_2d
from numpy import all as all_

from limix_math.linalg import qs_decomposition

from .transformation import DesignMatrixTrans
from ..inference import FastLMM as FastLMMCore
from ..func import maximize_scalar
from ..func import Learnable
from ..func import Variables
from ..func import Scalar
from ..func import FuncData

class FastLMM(Learnable, FuncData):
    def __init__(self, y, covariates, X=None, QS=None):
        self._logistic = Scalar(0.0)
        Learnable.__init__(self, Variables(logistic=self._logistic))
        FuncData.__init__(self)

        assert (X is None) != (QS is None)
        if not all_(isfinite(y)):
            raise ValueError("There are non-finite values in the phenotype.")

        if QS is None:
            self._trans = DesignMatrixTrans(X)
            X = self._trans.transform(X)
            QS = qs_decomposition(X)
            self._X = X

        self._flmmc = FastLMMCore(y, covariates, QS[0][0], QS[0][1], QS[1][0])

        self._y = y
        self._covariates = covariates
        self._QS = QS

        self._beta = None
        self._genetic_variance = None
        self._noise_variance = None

    def _delta(self):
        v = clip(self._logistic.value, -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, 1e-5, 1-1e-5)

    @property
    def genetic_variance(self):
        return self._genetic_variance

    @property
    def noise_variance(self):
        return self._noise_variance

    @property
    def beta(self):
        return self._beta

    @property
    def mean(self):
        return self._flmmc.mean

    def learn(self):
        maximize_scalar(self)

        delta = self._delta()
        self._flmmc.delta = delta

        beta = self._flmmc.beta
        scale = self._flmmc.scale

        self._genetic_variance = scale * (1 - delta)
        self._noise_variance = scale * delta
        self._beta = beta

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
