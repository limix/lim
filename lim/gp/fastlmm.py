from __future__ import division

from numpy import log
from numpy import exp
from numpy import pi
from numpy import sum
from numpy import empty
from numpy import logaddexp
from numpy.linalg import solve
from numpy.linalg import slogdet

from ..func.optimize.brent import maximize as maximize_scalar
from ..func.optimize.tnc import maximize as maximize_array
from ..genetics import eigen_design_matrix

from ..random import GPSampler

def _logsumexp_as(arr, scalar, out):
    for i in range(arr.shape[0]):
        out[i] = logaddexp(arr[i], scalar)

def _logsumexp_aa(arr0, arr1, out):
    for i in range(arr0.shape[0]):
        out[i] = logaddexp(arr0[i], arr1[i])

class FastLMM(object):
    def __init__(self, y, X):
        self._y = y
        self._X = X
        QS = eigen_design_matrix(X)
        self._Q = QS[0]
        self._S = QS[1]
        self._lS0 = log(self._S[0])

        self._offset = 0.0
        self._logscale = 0.0
        self._logdelta = 0.0

        self._logdiagi_aux0 = empty(len(self._S[0]))

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o):
        self._offset = o

    @property
    def scale(self):
        return exp(self._logscale)

    @scale.setter
    def scale(self, s):
        self._logscale = log(s)

    @property
    def delta(self):
        return exp(self._logdelta)

    @delta.setter
    def delta(self, d):
        self._logdelta = log(d)

    def _Qty(self):
        Q = self._Q
        return (Q[0].T.dot(self._y), Q[1].T.dot(self._y))

    def _Qtones(self):
        Q = self._Q
        return (Q[0].sum(0), Q[1].sum(0))

    def _Qtoffset(self):
        Qto = self._Qtones()
        return (Qto[0] * self._offset, Qto[1] * self._offset)

    def _diag(self):
        return (self._S[0] + exp(self._logdelta), exp(self._logdelta))

    def _logdiagi(self):
        lS0 = self._lS0
        out = empty(lS0.shape[0])
        out = self._logdiagi_aux0
        _logsumexp_as(lS0, self._logdelta, out)
        return (-out, -self._logdelta)

    def _logdet(self):
        d = self._diag()
        n = len(self._y)
        return n * self._logscale + log(d[0]).sum() + (n-len(d[0])) * log(d[1])

    def lml(self):
        a = self._Qty()
        b = self._Qtoffset()
        diff = (a[0] - b[0], a[1] - b[1])

        ldi = self._logdiagi()
        si = -self._logscale
        ymKiym0 = sum(exp(si + ldi[0]) * diff[0] * diff[0])
        ymKiym1 = sum(exp(si + ldi[1]) * diff[1] * diff[1])

        ymKiym = ymKiym0 + ymKiym1
        logdet = self._logdet()

        n = len(self._y)
        return - (logdet + ymKiym + n * log(2*pi)) / 2

    # def lml_gradient(self):
    #     grad_cov = self._lml_gradient_cov()
    #     grad_mean = self._lml_gradient_mean()
    #     return grad_cov + grad_mean
    #
    # def _lml_gradient_mean(self):
    #     mean = self._mean
    #
    #     vars_ = mean.variables().select(fixed=False)
    #
    #     Kiym = self._Kim()
    #
    #     g = []
    #     for i in range(len(vars_)):
    #         dm = mean.data('learn').gradient()[i]
    #         g.append(dm.dot(Kiym))
    #     return g
    #
    # def _lml_gradient_cov(self):
    #     cov = self._cov
    #
    #     vars_ = cov.variables().select(fixed=False)
    #     K = cov.data('learn').value()
    #     Kiym = self._Kim()
    #
    #     g = []
    #     for i in range(len(vars_)):
    #         dK = self._cov.data('learn').gradient()[i]
    #         g.append(- solve(K, dK).diagonal().sum()
    #                  + Kiym.dot(dK.dot(Kiym)))
    #     return [gi / 2 for gi in g]
    #
    # def _Kim(self):
    #     m = self._mean.data('learn').value()
    #     K = self._cov.data('learn').value()
    #     return solve(K, self._y - m)
    #
    # def variables(self):
    #     v0 = self._mean.variables().select(fixed=False)
    #     v1 = self._cov.variables().select(fixed=False)
    #     return merge_variables(dict(mean=v0, cov=v1))
    #
    # def learn(self):
    #     self.value = lambda: self.lml()
    #     self.gradient = lambda: self.lml_gradient()
    #
    #     if len(self.variables()) == 0:
    #         return
    #     elif len(self.variables()) == 1:
    #         maximize_scalar(self)
    #     else:
    #         maximize_array(self)
    #
    # def predict(self):
    #     y = self._y
    #     mean = self._mean
    #     cov = self._cov
    #
    #     m_p = mean.data('predict').value()
    #     _Kim = self._Kim()
    #     K_pp = cov.data('predict').value()
    #
    #     K_lp = cov.data('learn_predict').value()
    #
    #     est_mean = m_p + K_lp.T.dot(_Kim)
    #
    #     return est_mean

if __name__ == '__main__':
    import numpy as np
    import logging
    logging.basicConfig(level=logging.INFO)
    np.random.seed(100)
    n = 5
    y = np.random.randint(0, 2, n)
    X = np.random.randn(n, 300)
    flmm = FastLMM(y, X)
    flmm.lml()
