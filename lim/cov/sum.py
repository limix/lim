import numpy as np

from ..func import LearnableReduce
from ..func import merge_variables
from ..func import BiFuncReduce

class SumCov(LearnableReduce, BiFuncReduce):
    def __init__(self, covariances):
        self._covariances = [c for c in covariances]
        import ipdb; ipdb.set_trace()
        LearnableReduce.__init__(self, 'sum', self._covariances)
        BiFuncReduce.__init__(self)

    def value(self, x0s, x1s):
        c = self._covariances
        r = np.add.reduce([c[i].value(x0s[i], x1s[i]) for i in range(len(c))])
        return r

    def _get_functions(self):
        return [c for c in self._covariances]
