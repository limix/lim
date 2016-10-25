from __future__ import division

from numpy import log
from numpy import pi
from numpy import var
from numpy.linalg import solve
from numpy.linalg import slogdet

from scipy.stats import multivariate_normal

from limix_math.linalg import sum2diag

from optimix import merge_variables
from optimix import maximize_scalar
from optimix import maximize
from optimix import as_data_function
from optimix import Composite

from ..math import epsilon


class SlowLMM(Composite):

    def __init__(self, y, mean, cov):
        if var(y) < 1e-8:
            raise ValueError("The phenotype variance is too low: %e." % var(y))

        super(SlowLMM, self).__init__(mean=mean, cov=cov)
        self._y = y
        self._cov = cov
        self._mean = mean

    def lml(self):
        y = self._y
        mean = self._mean
        cov = self._cov
        Kiym = self._Kim()

        ym = y - as_data_function(mean).value()

        (s, logdet) = slogdet(as_data_function(cov).value())
        assert s == 1.

        n = len(y)
        return - (logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    def _lml_gradient_mean(self):
        mean = self._mean

        vars_ = mean.variables().select(fixed=False)

        Kiym = self._Kim()

        g = []
        for i in range(len(vars_)):
            dm = as_data_function(mean).gradient()[i]
            g.append(dm.T.dot(Kiym))
        return g

    def _lml_gradient_cov(self):
        cov = self._cov

        vars_ = cov.variables().select(fixed=False)
        K = as_data_function(cov).value()
        Kiym = self._Kim()

        g = []
        for i in range(len(vars_)):
            dK = as_data_function(cov).gradient()[i]
            g.append(- solve(K, dK).diagonal().sum() +
                     Kiym.dot(dK.dot(Kiym)))
        return [gi / 2 for gi in g]

    def _Kim(self):
        m = as_data_function(self._mean).value()
        K = as_data_function(self._cov).value()
        return solve(K, self._y - m)

    def variables(self):
        v0 = self._mean.variables().select(fixed=False)
        v1 = self._cov.variables().select(fixed=False)
        return merge_variables(dict(mean=v0, cov=v1))

    def value(self, mean, cov):
        ym = self._y - mean
        Kiym = solve(cov, ym)


        (s, logdet) = slogdet(cov)
        assert s == 1.

        n = len(self._y)
        return - (logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    def gradient(self, mean, cov, gmean, gcov):
        grad_cov = self._lml_gradient_cov()
        grad_mean = self._lml_gradient_mean()
        return grad_cov + grad_mean

    def learn(self):
        if len(self.variables()) == 0:
            return
        elif len(self.variables()) == 1:
            maximize_scalar(self)
        else:
            maximize(self)

    def predict(self):
        mean = self._mean
        cov = self._cov

        m_p = mean.data('predict').value()
        _Kim = self._Kim()
        K_pp = cov.data('predict').value()

        K_lp = cov.data('learn_predict').value()

        emean = m_p + K_lp.T.dot(_Kim)
        K = as_data_function(cov).value()
        ecov = K_pp - K_lp.T.dot(solve(K, K_lp))

        return SlowLMMPredictor(emean, ecov)


class SlowLMMPredictor(object):

    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._mvn = multivariate_normal(mean, sum2diag(cov, epsilon.small))

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
