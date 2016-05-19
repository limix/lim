import numpy as np

from ..func import Learnable
from ..func import merge_variables
from ..func import BiFuncReduce

class SumCov(Learnable, BiFuncReduce):
    def __init__(self, covariances):
        self._covariances = []
        for c in covariances:
            self._covariances.append(c)

        vars_list = [c.variables() for c in self._covariances]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd['sum[%d]' % i] = vs
        variables = merge_variables(vd)

        Learnable.__init__(self, variables)
        BiFuncReduce.__init__(self)

    def value(self, x0s, x1s):
        c = self._covariances
        r = np.add.reduce([c[i].value(x0s[i], x1s[i]) for i in range(len(c))])
        return r

    def _get_functions(self):
        return [c for c in self._covariances]
