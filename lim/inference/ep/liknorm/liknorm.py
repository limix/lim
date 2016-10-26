from __future__ import division

from numba import cffi_support as _cffi_support
from numpy import ndarray

from . import _liknorm_ffi

_cffi_support.register_module(_liknorm_ffi)


def ptr(a):
    if isinstance(a, ndarray):
        return _liknorm_ffi.ffi.cast("double *", a.ctypes.data)
    return a


class LikNormMoments(object):

    def __init__(self, nintervals):
        super(LikNormMoments, self).__init__()

        _liknorm_ffi.lib.initialize(nintervals)

    def binomial(self, k, n, eta, tau, log_zeroth, mean, variance):
        size = len(k)
        _liknorm_ffi.lib.binomial_moments(size, ptr(k), ptr(n),
                                          ptr(eta), ptr(tau),
                                          ptr(log_zeroth), ptr(mean),
                                          ptr(variance))

    def poisson(self, k, eta, tau, log_zeroth, mean, variance):
        size = len(k)
        _liknorm_ffi.lib.poisson_moments(size, ptr(k), ptr(eta), ptr(tau),
                                         ptr(log_zeroth), ptr(mean),
                                         ptr(variance))

    def exponential(self, x, eta, tau, log_zeroth, mean, variance):
        size = len(x)
        _liknorm_ffi.lib.exponential_moments(size, ptr(x), ptr(eta), ptr(tau),
                                             ptr(log_zeroth), ptr(mean),
                                             ptr(variance))

    def destroy(self):
        _liknorm_ffi.lib.destroy()
