from numpy.random import RandomState
from numpy_sugar.random import multivariate_normal


class RegGPSampler(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = self._mean.feed('sample').value()
        K = self._cov.feed('sample').value()
        return multivariate_normal(m, K, random_state)
