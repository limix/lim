from __future__ import division

# from numba import jitclass
from numpy import dot
from numpy import sum
from numpy import empty
from numpy import log

# spec = [
#     ('value', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
# ]

# @jitclass
class FastLMM(object):
    def __init__(self, y, Q0, Q1, S0):
        self.n = y.shape[0]
        self.p = self.n - S0.shape[0]
        self.S0 = S0
        self.diag0 = empty(Q0.shape[1])
        self.diag1 = 0.
        self.m = empty(self.n)

        self._offset = 0.0
        self._scale = 1.0
        self._delta = 0.5
        self._lml = 0.0

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

        self._valid_update = False

    @property
    def offset(self):
        return self._offset

    @property
    def scale(self):
        return self._scale

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._valid_update = False
        self._delta = delta

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
        self._offset = nominator / denominator

    def _update_scale(self):
        o = self._offset
        o2 = o**2
        self._scale = (self.a1 - 2 * self.b1 * o + self.c1 * o2 +
                      self.a0 - 2 * self.b0 * o + self.c0 * o2) / self.n

    def _update_diags(self):
        self.diag0[:] = self.S0
        self.diag0 *= (1 - self._delta)
        self.diag0 += self._delta
        self.diag1 = self._delta

    def _update(self):
        if self._valid_update:
            return

        self._update_diags()
        self._update_joints()
        self._update_offset()
        self._update_scale()

        self._valid_update = True

    def lml(self):
        if self._valid_update:
            return self._lml

        self._update()

        n = self.n
        p = self.p
        LOG2PI = 1.837877066409345339081937709124758839607238769531250

        self._lml  = - n * LOG2PI - n - n * log(self._scale)
        self._lml +=  - sum(log(self.diag0)) - p * log(self.diag1)
        self._lml /= 2
        return self._lml
