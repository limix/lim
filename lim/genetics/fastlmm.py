from __future__ import division

from numpy import exp

from .decomposition import eigen_design_matrix
from ..inference import FastLMM as FastLMMCore
from ..func.optimize.brent import maximize
from ..func import Learnable
from ..func import Variables
from ..func import Scalar
from ..func import FuncData

class FastLMM(Learnable, FuncData):
    def __init__(self, y, X):
        self._logistic = Scalar(0.0)
        Learnable.__init__(self, Variables(logistic=self._logistic))
        FuncData.__init__(self)
        QS = eigen_design_matrix(X)
        self._flmmc = FastLMMCore(y, QS[0][0], QS[0][1], QS[1][0])
        self._genetic_variance = None
        self._noise_variance = None
        self._offset = None

    def _delta(self):
        return 1 / (1 + exp(self._logistic.value))

    @property
    def genetic_variance(self):
        return self._genetic_variance

    @property
    def noise_variance(self):
        return self._noise_variance

    @property
    def offset(self):
        return self._offset

    def learn(self):
        maximize(self)

        delta = self._delta()
        self._flmmc.delta = delta

        offset = self._flmmc.offset
        scale = self._flmmc.scale

        self._genetic_variance = scale * (1 - delta)
        self._noise_variance = scale * delta
        self._offset = offset

    def value(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()
