from __future__ import division

from numpy import dot
from numpy import empty
from numpy import log
from numpy import var

from scipy.stats import multivariate_normal

from limix_math.linalg import sum2diag

class FastLMM(object):
    def __init__(self, y, Q0, Q1, S0):

        if var(y) < 1e-8:
            raise ValueError("The phenotype variance is too low: %e." % var(y))

        self._n = y.shape[0]
        self._p = self._n - S0.shape[0]
        self._S0 = S0
        self._diag0 = S0 * 0.5 + 0.5
        self._diag1 = 0.5
        self._m = empty(self._n)
        self._Q0 = Q0
        self._Q1 = Q1

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

        self._yTQ0 = dot(y.T, Q0)
        self._yTQ0_2x = self._yTQ0 ** 2

        self._yTQ1 = dot(y.T, Q1)
        self._yTQ1_2x = self._yTQ1 ** 2

        self._oneTQ0 = Q0.sum(0)
        self._oneTQ0_2x = self._oneTQ0 ** 2

        self._oneTQ1 = Q1.sum(0)
        self._oneTQ1_2x = self._oneTQ1 ** 2

        self._yTQ0_oneTQ0 = self._yTQ0 * self._oneTQ0
        self._yTQ1_oneTQ1 = self._yTQ1 * self._oneTQ1

        self._valid_update = 0
        self.__Q0tymD0 = None
        self.__Q1tymD1 = None

    def _Q0tymD0(self):
        if self.__Q0tymD0 is None:
            Q0tym = self._yTQ0 - self._oneTQ0 * self._offset
            self.__Q0tymD0 = Q0tym / self._diag0
        return self.__Q0tymD0

    def _Q1tymD1(self):
        if self.__Q1tymD1 is None:
            Q1tym = self._yTQ1 - self._oneTQ1 * self._offset
            self.__Q1tymD1 = Q1tym / self._diag1
        return self.__Q1tymD1

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
        self.__Q0tymD0 = None
        self.__Q1tymD1 = None
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

    def predict(self, Cp, Cpp):
        delta = self.delta

        diag0 = self._diag0
        diag1 = self._diag1

        CpQ0 = Cp.dot(self._Q0)
        CpQ1 = Cp.dot(self._Q1)

        mean  = self._offset + (1-delta) * CpQ0.dot(self._Q0tymD0())
        mean += (1-delta) * CpQ1.dot(self._Q1tymD1())

        cov = sum2diag(Cpp * (1-self.delta), self.delta)
        cov -= (1-delta)**2 * CpQ0.dot((CpQ0 / diag0).T)
        cov -= (1-delta)**2 * CpQ1.dot((CpQ1 / diag1).T)
        cov *= self.scale

        return FastLMMPredictor(mean, cov)

class FastLMMPredictor(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._mvn = multivariate_normal(mean, cov)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def pdf(self, y):
        return self._mvn.pdf(y)

    def logpdf(self, y):
        return self._mvn.logpdf(y)
