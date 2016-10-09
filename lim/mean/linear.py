from numpy import full
from numpy import zeros
from numpy import ascontiguousarray

from ..func import Learnable
from ..func import Vector
from ..func import FuncData
from ..func import Variables


class LinearMean(Learnable, FuncData):

    def __init__(self, size):
        self._effsizes = Vector(zeros(size))
        Learnable.__init__(self, Variables(effsizes=self._effsizes))
        FuncData.__init__(self)

    @property
    def effsizes(self):
        return self._effsizes.value

    @effsizes.setter
    def effsizes(self, effsizes):
        effsizes = ascontiguousarray(effsizes, float)
        if effsizes.shape != self._effsizes.shape:
            raise ValueError("could not broadcast input array" +
                             " from shape %s into shape %s" %
                             (effsizes.shape, self._effsizes.shape))
        self._effsizes.value = effsizes

    def value(self, x):
        return x.dot(self._effsizes)

    def derivative_effsizes(self, x):
        return x
