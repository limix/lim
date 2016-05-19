from numpy import exp
from numpy import log

from ..func import Learnable
from ..func import Scalar

class LinearCov(Learnable):
    def __init__(self):
        self._logscale = Scalar(0.0)
        Learnable.__init__(self, logscale=self._logscale)

    @property
    def scale(self):
        return exp(self._logscale.value)

    @scale.setter
    def scale(self, scale):
        self._logscale.value = log(scale)

    def value(self, x0, x1):
        return self.scale * x0.dot(x1.T)

    def derivative_logscale(self, x0, x1):
        return self.scale * x0.dot(x1.T)
