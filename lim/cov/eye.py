from numpy import eye
from numpy import exp
from numpy import log
from numpy import array

from ..func import Learnable
from ..func import Variables
from ..func import Scalar
from ..func import FuncData
from ..util.scalar import isnumber

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

        # if isnumber(x0):
        #     assert isnumber(x1)
        #     return self.scale * (x0 is x1)
        # if x0.ndim == 1:
        #     return array([self.scale * (x0 is x1)])
        # elif x0.ndim == 2:
        #     return self.scale * (x0 is x1) * eye(x0.shape[0])
        # assert False

    def derivative_logscale(self, x0, x1):
        return self.value(x0, x1)
