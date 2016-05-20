from __future__ import division

from numpy import log
from numpy import exp
from numpy import pi
from numpy import sum
from numpy import empty
from numpy import logaddexp
from numpy.linalg import solve
from numpy.linalg import slogdet

from numba import jit

from ..func.optimize.brent import maximize as maximize_scalar
from ..func.optimize.tnc import maximize as maximize_array
from ..genetics import eigen_design_matrix

from ..random import GPSampler

# def _logsumexp_as(arr, scalar, out):
#     for i in range(arr.shape[0]):
#         out[i] = logaddexp(arr[i], scalar)
#
# def _logsumexp_aa(arr0, arr1, out):
#     for i in range(arr0.shape[0]):
#         out[i] = logaddexp(arr0[i], arr1[i])

def logaddexp(x, y):
    LOGE2 = 0.693147180559945286226763982995180413126945495605468750
    if x == y:
        return x + LOGE2
    tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    return tmp

def _log_diag_inv0(lS0, ldelta, ldelta1, out):
    for i in range(lS0.shape[0]):
        out[i] = -logaddexp(lS0[i] + ldelta1, ldelta)

from numpy import dot
from numpy import abs
from numpy import sum

def _offset(Qty0, Qty1, Qtones01, Qtones11, Qtones02, Qtones12, di0, di1):
    a = dot(Qtones02, di0)
    b = dot(Qtones12, di1)
    denom = b - a
    if abs(denom) < 1e-10:
        return 0.0

    a = dot(Qty0, Qtones01) * di0
    b = dot(Qty1, Qtones11) * di1
    nom = b - a

    return nom/denom

def _scale(diff02, diff12, di0, di1):
    n = diff02.shape[0]
    ymKiym0 = dot(di0, diff02)
    ymKiym1 = di1 * sum(diff12)
    return (ymKiym0 + ymKiym1)/n

def _diff(Qty, Qtones, offset, out):
    for i in range(out.shape[0]):
        out[i] = Qty[i] - offset * Qtones[i]

def _logdet(ldi0, ldi1, logscale):
    # ldi = logdiagi
    n = ldi0.shape[0]
    return n * logscale - sum(ldi0) - (n-ldi0.shape[0]) * ldi1

class FastLMM(object):
    def __init__(self, y, X):
        self._y = y
        self._X = X
        QS = eigen_design_matrix(X)
        self._Q = QS[0]
        self._S = QS[1]
        self._lS0 = log(self._S[0])

        self._logdelta = 0.0

        self._logdiagi_aux0 = empty(len(self._S[0]))

    def _Qty(self):
        Q = self._Q
        return (Q[0].T.dot(self._y), Q[1].T.dot(self._y))

    def _Qtones(self):
        Q = self._Q
        return (Q[0].sum(0), Q[1].sum(0))

    # def offset(self):
    #     Qty = self._Qty()
    #     Qtones = self._Qtones()
    #     ldi = self._logdiagi()
    #     n = len(self._y)
    #
    #     a = sum(Qtones[0] * Qtones[0] * exp(ldi[0]))
    #     b = sum(Qtones[1] * Qtones[1] * exp(ldi[1]))
    #     denom = b - a
    #     if abs(denom) < 1e-10:
    #         return 0.0
    #
    #     a = sum(Qty[0] * Qtones[0] * exp(ldi[0]))
    #     b = sum(Qty[1] * Qtones[1] * exp(ldi[1]))
    #     nom = b - a
    #
    #     return nom/denom

    def scale(self):
        return exp(self._logscale())

    def _logscale(self):
        diff = self._diff()

        ldi = self._logdiagi()

        ymKiym0 = sum(exp(ldi[0]) * diff[0] * diff[0])
        ymKiym1 = sum(exp(ldi[1]) * diff[1] * diff[1])

        ymKiym = ymKiym0 + ymKiym1
        return log(ymKiym / len(self._y))

    def _Qtoffset(self):
        Qto = self._Qtones()
        return (Qto[0] * self.offset(), Qto[1] * self.offset())

    def _logdiagi(self):
        lS0 = self._lS0
        out = self._logdiagi_aux0
        _logsumexp_as(lS0 + log(1-self.delta), self._logdelta, out)
        return (-out, -self._logdelta)

    def _logdet(self, logdiagi=None):
        if logdiagi is None:
            ldi = self._logdiagi()
        else:
            ldi = logdiagi
        n = len(self._y)
        return n * self._logscale() - ldi[0].sum() - (n-len(ldi[0])) * ldi[1]

    def _diff(self):
        a = self._Qty()
        b = self._Qtoffset()
        return (a[0] - b[0], a[1] - b[1])

    def lml(self):
        # Qty = self._Qty
        # Qtones = self._Qtones
        # lS0 = self._lS0
        # ldelta = self.logdelta
        # ldelta = self.logdelta1
        # (Qty0, Qty1, Qtones01, Qtones11, Qtones02, Qtones12, di0, di1)
        logdelta = self._logdelta
        logdelta1 = log(1 - self.delta)

        _log_diag_inv0(lS0, logdelta, logdelta1, ldi0)
        di1 = - self.logdelta
        offset(Qty0, Qty1, Qtones01, Qtones11, Qtones02, Qtones12, log(di0), di1)
        _diff(Qty[0], Qtones[0], offset, diff0)
        _diff(Qty[1], Qtones[1], offset, diff1)

        # si = -self._logscale()
        # ymKiym0 = sum(exp(si + ldi[0]) * diff[0] * diff[0])
        # ymKiym1 = sum(exp(si + ldi[1]) * diff[1] * diff[1])

        # ymKiym = ymKiym0 + ymKiym1
        logdet = self._logdet(logdiagi=ldi)

        scale = _scale(diff02, diff12, di0, di1)

        n = len(self._y)
        return - (logdet + scale*n + n * log(2*pi)) / 2

    def learn(self):
        nsteps = 1000
        step = 1/(nsteps+1)
        self.delta = step
        best_step = 0
        lml0 = self.lml()
        for i in range(1, nsteps):
            self.delta = i * step + step
            lml1 = self.lml()
            if lml1 > lml0:
                lml0 = lml1
                best_step = i
        self.delta = best_step * step + step
