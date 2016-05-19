from numpy import full
from numpy import ones

from ..func import Learnable
from ..func import Scalar
from ..func import FuncData
from ..func import Variables

class OffsetMean(Learnable, FuncData):
    def __init__(self):
        self._offset = Scalar(1.0)
        Learnable.__init__(self, Variables(offset=self._offset))
        FuncData.__init__(self)

    @property
    def offset(self):
        return self._offset.value

    @offset.setter
    def offset(self, offset):
        self._offset.value = offset

    def value(self, size):
        return full(size, self.offset)

    def derivative_offset(self, size):
        return ones(size)
