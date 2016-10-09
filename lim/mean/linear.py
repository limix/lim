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
        self._effsizes.value = ascontiguousarray(effsizes, float)

    def value(self, x):
        return x.dot(self._effsizes)

    def derivative_offset(self, x):
        return x
