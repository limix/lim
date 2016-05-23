from numpy import full
from numpy.random import RandomState

from limix_math.linalg import sum2diag_inplace

from ..genetics import DesignMatrixTrans

class FastLMMSampler(object):
    def __init__(self, offset, scale, delta, X):
        t = DesignMatrixTrans(X)
        X = t.transform(X)
        self._offset = offset
        self._cov = X.dot(X.T) * (1-delta)
        sum2diag_inplace(self._cov, delta)
        self._cov *= scale

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        K = self._cov
        o = full(K.shape[0], self._offset)
        return random_state.multivariate_normal(o, K)
