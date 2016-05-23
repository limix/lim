from numpy import exp
from numpy import log

from ..func import Learnable
from ..func import Variables
from ..func import Scalar
from ..func import FuncData

class EyeCov(Learnable, FuncData):
    def __init__(self):
        self._logscale = Scalar(0.0)
        Learnable.__init__(self, Variables(logscale=self._logscale))
        FuncData.__init__(self)

    @property
    def scale(self):
        return exp(self._logscale.value)

    @scale.setter
    def scale(self, scale):
        self._logscale.value = log(scale)

    def value(self, x0, x1):
        return self.scale * (x0 == x1)

    def derivative_logscale(self, x0, x1):
        return self.value(x0, x1)
