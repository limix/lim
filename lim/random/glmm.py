from __future__ import division

from numpy.random import RandomState


class GLMMSampler(object):

    def __init__(self, lik, mean, cov):
        self._lik = lik
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = self._mean.feed('sample').value()
        K = self._cov.feed('sample').value()
        u = random_state.multivariate_normal(m, K)

        return self._lik.sample(u, random_state)
