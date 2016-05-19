import numpy as np

from ..func import Learnable
from ..func import merge_variables

def _collect_variables(my_name, covariances):
    v = dict()
    for (i, c) in enumerate(covariances):
        your_name = c.__class__.__name__
        pre = '[%d](' % i
        suf = ')'
        v[my_name + pre + your_name + suf] = c.variables()
    return v

class SumCov(Learnable):
    def __init__(self, covariances):
        self._covariances = []
        for c in covariances:
            self._covariances.append(c)

        my_name = self.__class__.__name__
        v = _collect_variables(my_name, self._covariances)
        Learnable.__init__(self, **v)

    def value(self):
        return np.add.reduce([c.value() for c in self._covariances])
