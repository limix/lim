from __future__ import division

from numpy import dot
from numpy import empty
from numpy import log

# from numba import jit
# from numba import jitclass
# from numba import float64
# from numba import int64
# from numba import boolean

# spec = [
#     ('n', int64),
#     ('p', int64),
#     ('S0', float64[:]),
#     ('diag0', float64[:]),
#     ('diag1', float64),
#     ('m', float64[:]),
#     ('offset', float64),
#     ('scale', float64),
#     ('delta_', float64),
#     ('lml_', float64),
#     ('yTQ0_2x', float64[:]),
#     ('yTQ1_2x', float64[:]),
#     ('oneTQ0_2x', float64[:]),
#     ('oneTQ1_2x', float64[:]),
#     ('yTQ0_oneTQ0', float64[:]),
#     ('yTQ1_oneTQ1', float64[:]),
#     ('valid_update_', int64),
# ]

class FastLMM(object):
    def __init__(self, y, Q0, Q1, S0):
        self._n = y.shape[0]
        self._p = self._n - S0.shape[0]
        self._S0 = S0
        self._diag0 = empty(Q0.shape[1])
        self._diag1 = 0.0
        self._m = empty(self._n)

        self._offset = 0.0
        self._scale = 1.0
        self._delta = 0.5
        self._lml = 0.0

        self.a1 = 0.0
        self.b1 = 0.0
        self.c1 = 0.0

        self.a0 = 0.0
        self.b0 = 0.0
        self.c0 = 0.0

        yTQ0 = dot(y.T, Q0)
        self._yTQ0_2x = yTQ0 ** 2

        yTQ1 = dot(y.T, Q1)
        self._yTQ1_2x = yTQ1 ** 2

        oneTQ0 = Q0.sum(0)
        self._oneTQ0_2x = oneTQ0 ** 2

        oneTQ1 = Q1.sum(0)
        self._oneTQ1_2x = oneTQ1 ** 2

        self._yTQ0_oneTQ0 = yTQ0 * oneTQ0
        self._yTQ1_oneTQ1 = yTQ1 * oneTQ1

        self._valid_update = 0

    @property
    def scale(self):
        return self._scale

    @property
    def offset(self):
        return self._offset

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._valid_update = 0
        self._delta = delta

    def _update_joints(self):
        yTQ0_2x = self._yTQ0_2x
        yTQ1_2x = self._yTQ1_2x

        oneTQ0_2x = self._oneTQ0_2x
        oneTQ1_2x = self._oneTQ1_2x

        yTQ0_oneTQ0 = self._yTQ0_oneTQ0
        yTQ1_oneTQ1 = self._yTQ1_oneTQ1

        self.a1 = yTQ1_2x.sum() / self._diag1
        self.b1 = yTQ1_oneTQ1.sum() / self._diag1
        self.c1 = oneTQ1_2x.sum() / self._diag1

        self.a0 = (yTQ0_2x / self._diag0).sum()
        self.b0 = (yTQ0_oneTQ0 / self._diag0).sum()
        self.c0 = (oneTQ0_2x / self._diag0).sum()

    def _update_offset(self):
        nominator = self.b1 - self.b0
        denominator = self.c1 - self.c0
        self._offset = nominator / denominator

    def _update_scale(self):
        o = self._offset
        o2 = o**2
        self._scale = (self.a1 - 2 * self.b1 * o + self.c1 * o2 +
                      self.a0 - 2 * self.b0 * o + self.c0 * o2) / self._n

    def _update_diags(self):
        self._diag0[:] = self._S0
        self._diag0 *= (1 - self._delta)
        self._diag0 += self._delta
        self._diag1 = self._delta

    def _update(self):
        if self._valid_update:
            return

        self._update_diags()
        self._update_joints()
        self._update_offset()
        self._update_scale()

        self._valid_update = 1

    def lml(self):
        if self._valid_update:
            return self._lml

        self._update()

        n = self._n
        p = self._p
        LOG2PI = 1.837877066409345339081937709124758839607238769531250

        self._lml  = - n * LOG2PI - n - n * log(self._scale)
        self._lml +=  - sum(log(self._diag0)) - p * log(self._diag1)
        self._lml /= 2
        return self._lml
