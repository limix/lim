from numpy import add

from ..func import LearnableReduce
from ..func import FuncDataReduce

class SumCov(LearnableReduce, FuncDataReduce):
    def __init__(self, covariances):
        self._covariances = [c for c in covariances]
        LearnableReduce.__init__(self, self._covariances, 'sum')
        FuncDataReduce.__init__(self, self._covariances)

    def value_reduce(self, values):
        return add.reduce(values)

    def derivative_reduce(self, derivatives):
        return add.reduce(derivatives)
