from numpy.random import RandomState

class RegGPSampler(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = self._mean.data('sample').value()
        K = self._cov.data('sample').value()
        return random_state.multivariate_normal(m, K)
