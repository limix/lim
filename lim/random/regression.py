from numpy.random import RandomState

from optimix import as_data_function


class RegGPSampler(object):

    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = as_data_function(self._mean, 'sample').value()
        K = as_data_function(self._cov, 'sample').value()
        return random_state.multivariate_normal(m, K)
