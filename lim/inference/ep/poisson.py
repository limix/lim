from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import (all, asarray, clip, exp, full, isfinite, isscalar, log,
                   ones, pi, set_printoptions, sqrt)
from numpy.linalg import lstsq

from limix_math import issingleton

from ...inference import FastLMM
from limix_math import issingleton
from .liknorm import LikNormMoments

from .ep import EP


class PoissonEP(EP):
    r"""Poisson EP inference.

    Let :math:`\lambda_i` be the rate of occurrence for the i-th individual and

    .. math::

        p(y_i | \lambda_i) = \frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!}

    be the Poisson likelihood, where :math:`y_i` is the number of occurrences.
    The marginal likelihood is

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(\lambda_i)=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    where

    .. math::

        g(x) = \log x

    is the canonical link function.

    Args:
        noccurrences (array_like): Array of :math:`y_i \in \{0, 1, \dots\}`.
        M (array_like): :math:`\mathrm M` covariates.
        Q0 (array_like): :math:`\mathrm Q_0` of the eigendecomposition.
        Q1 (array_like): :math:`\mathrm Q_1` of the eigendecomposition.
        S0 (array_like): :math:`\mathrm S_0` of the eigendecomposition.
        Q0S0Q0t (array_like): :math:`\mathrm Q_0 \mathrm S_0
                        \mathrm Q_0^{\intercal}` in case this has already
                        been computed. Defaults to `None`.
    """

    def __init__(self, noccurrences, M, Q0, Q1, S0,
                 Q0S0Q0t=None):
        super(PoissonEP, self).__init__(M, Q0, S0, True, QSQt=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        noccurrences = asarray(noccurrences, float)

        self._noccurrences = noccurrences

        if issingleton(noccurrences):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not all(isfinite(noccurrences)):
            raise ValueError("There are non-finite numbers in phenotype.")

        assert noccurrences.shape[0] == M.shape[
            0], 'Number of individuals mismatch.'
        assert noccurrences.shape[0] == Q0.shape[
            0], 'Number of individuals mismatch.'
        assert noccurrences.shape[0] == Q1.shape[
            0], 'Number of individuals mismatch.'

        self._Q1 = Q1

        self._moments = LikNormMoments(350)
        self.initialize()

    @property
    def genetic_variance(self):
        r"""Returns :math:`\sigma_b^2`."""
        return self.sigma2_b

    @property
    def environmental_variance(self):
        r"""Returns :math:`\sigma_{\epsilon}^2`."""
        return self.sigma2_epsilon

    @property
    def heritability(self):
        r"""Returns
:math:`\sigma_b^2/(\sigma_a^2+\sigma_b^2+\sigma_{\epsilon}^2)`."""
        total = self.genetic_variance + self.covariates_variance
        total += self.environmental_variance
        return self.genetic_variance / total

    def initialize(self):
        from scipy.stats import norm

        noccurrences = self._noccurrences

        latent = (noccurrences - noccurrences.mean()) / noccurrences.std()

        Q0 = self._Q
        Q1 = self._Q1
        S0 = self._S
        covariates = self._M
        flmm = FastLMM(latent, covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.environmental_variance
        h2 = gv / (gv + nv)
        h2 = clip(h2, 0.01, 0.9)

        mean = flmm.mean
        self._tbeta = lstsq(self._tM, full(len(noccurrences), mean))[0]

        self.delta = 1 - h2
        self.v = gv + nv

    def _tilted_params(self):
        noccurrences = self._noccurrences
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz

        self._moments.poisson(noccurrences, ceta,
                              ctau, lmom0, self._hmu, self._hvar)
