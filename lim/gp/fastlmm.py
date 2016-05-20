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

    def _logdiagi(self):
        lS0 = self._lS0
        out = self._logdiagi_aux0
        _logsumexp_as(lS0, self._logdelta, out)
        return (-out, -self._logdelta)

    def _logdet(self, logdiagi=None):
        if logdiagi is None:
            ldi = self._logdiagi()
        else:
            ldi = logdiagi
        n = len(self._y)
        return n * self._logscale - ldi[0].sum() - (n-len(ldi[0])) * ldi[1]

    def _diff(self):
        a = self._Qty()
        b = self._Qtoffset()
        return (a[0] - b[0], a[1] - b[1])

    def lml(self, logdiagi=None):
        diff = self._diff()

        if logdiagi is None:
            ldi = self._logdiagi()
        else:
            ldi = logdiagi

        si = -self._logscale
        ymKiym0 = sum(exp(si + ldi[0]) * diff[0] * diff[0])
        ymKiym1 = sum(exp(si + ldi[1]) * diff[1] * diff[1])

        ymKiym = ymKiym0 + ymKiym1
        logdet = self._logdet(logdiagi=ldi)

        n = len(self._y)
        return - (logdet + ymKiym + n * log(2*pi)) / 2

    def optimal_delta(self):
        nsteps = 100
        step = 1/(nsteps+1)
        logstep = log(step)

        lS0 = self._lS0
        out = self._logdiagi_aux0
        _logsumexp_as(lS0, logstep, out)
        lml0 = self.lml(logdiagi=(-out, -logstep))
        best_step = 0

        for i in range(1, nsteps):
            logdelta = log(i * step + step)
            _logsumexp_as(out, logstep, out)
            lml1 = self.lml(logdiagi=(-out, -logdelta))
            if lml1 > lml0:
                lml0 = lml1
                best_step = i

        self.delta = best_step * step + step

    def optimal_scale(self, logdiagi=None):
        diff = self._diff()

        if logdiagi is None:
            ldi = self._logdiagi()
        else:
            ldi = logdiagi

        ymKiym0 = sum(exp(ldi[0]) * diff[0] * diff[0])
        ymKiym1 = sum(exp(ldi[1]) * diff[1] * diff[1])

        ymKiym = ymKiym0 + ymKiym1
        self.scale = ymKiym / len(self._y)

    def optimal_offset(self):
        Qty = self._Qty()
        Qtones = self._Qtones()
        ldi = self._logdiagi()
        n = len(self._y)

        a = sum(Qtones[0] * Qtones[0] * exp(ldi[0]))
        b = sum(Qtones[1] * Qtones[1] * exp(ldi[1]))
        denom = b - a
        if abs(denom) < 1e-10:
            self._offset = 0.0
            return

        a = sum(Qty[0] * Qtones[0] * exp(ldi[0]))
        b = sum(Qty[1] * Qtones[1] * exp(ldi[1]))
        nom = b - a

        self._offset = nom/denom
