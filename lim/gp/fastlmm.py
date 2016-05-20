from __future__ import division

from numpy import dot
from numpy import sum
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
        self.n = y.shape[0]
        self.p = self.n - S0.shape[0]
        self.S0 = S0
        self.diag0 = empty(Q0.shape[1])
        self.diag1 = 0.0
        self.m = empty(self.n)

        self.offset = 0.0
        self.scale = 1.0
        self.delta_ = 0.5
        self.lml_ = 0.0

        yTQ0 = dot(y.T, Q0)
        self.yTQ0_2x = yTQ0 ** 2

        yTQ1 = dot(y.T, Q1)
        self.yTQ1_2x = yTQ1 ** 2

        oneTQ0 = Q0.sum(0)
        self.oneTQ0_2x = oneTQ0 ** 2

        oneTQ1 = Q1.sum(0)
        self.oneTQ1_2x = oneTQ1 ** 2

        self.yTQ0_oneTQ0 = yTQ0 * oneTQ0
        self.yTQ1_oneTQ1 = yTQ1 * oneTQ1

        self.valid_update_ = 0

    @property
    def delta(self):
        return self.delta_

    @delta.setter
    def delta(self, delta):
        self.valid_update_ = 0
        self.delta_ = delta

    def _update_joints(self):
        yTQ0_2x = self.yTQ0_2x
        yTQ1_2x = self.yTQ1_2x

        oneTQ0_2x = self.oneTQ0_2x
        oneTQ1_2x = self.oneTQ1_2x

        yTQ0_oneTQ0 = self.yTQ0_oneTQ0
        yTQ1_oneTQ1 = self.yTQ1_oneTQ1

        self.a1 = sum(yTQ1_2x) / self.diag1
        self.b1 = sum(yTQ1_oneTQ1) / self.diag1
        self.c1 = sum(oneTQ1_2x) / self.diag1

        self.a0 = sum(yTQ0_2x / self.diag0)
        self.b0 = sum(yTQ0_oneTQ0 / self.diag0)
        self.c0 = sum(oneTQ0_2x / self.diag0)

    def _update_offset(self):
        nominator = self.b1 - self.b0
        denominator = self.c1 - self.c0
        self.offset = nominator / denominator

    def _update_scale(self):
        o = self.offset
        o2 = o**2
        self.scale = (self.a1 - 2 * self.b1 * o + self.c1 * o2 +
                      self.a0 - 2 * self.b0 * o + self.c0 * o2) / self.n

    def _update_diags(self):
        self.diag0[:] = self.S0
        self.diag0 *= (1 - self.delta_)
        self.diag0 += self.delta_
        self.diag1 = self.delta_

    def _update(self):
        if self.valid_update_:
            return

        self._update_diags()
        self._update_joints()
        self._update_offset()
        self._update_scale()

        self.valid_update_ = 1

    def lml(self):
        if self.valid_update_:
            return self.lml_

        self._update()

        n = self.n
        p = self.p
        LOG2PI = 1.837877066409345339081937709124758839607238769531250

        self.lml_  = - n * LOG2PI - n - n * log(self.scale)
        self.lml_ +=  - sum(log(self.diag0)) - p * log(self.diag1)
        self.lml_ /= 2
        return self.lml_
