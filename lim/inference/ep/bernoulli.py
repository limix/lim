from __future__ import absolute_import, division, unicode_literals

import logging

import scipy.stats as st

from numpy import (all, asarray, clip, exp, full, isfinite, log, ones, pi,
                   sqrt)
from numpy.linalg import lstsq

from ...inference import FastLMM
from limix_math import issingleton
from .liknorm import LikNormMoments

from .ep import EP


class BernoulliEP(EP):
    r"""Bernoulli EP inference.

    Let :math:`p_i` be the probability of success for the i-th individual and

    .. math::

        p(y_i | p_i) = p_i^{y_i} (1-p_i)^{1-y_i}

    be the Bernoulli likelihood. It assumes the marginal likelihood

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(p_i)=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    where

    .. math::

        g(x) = \log \left(\frac{x}{1-x}\right)

    is the Logit link function.

    We set :math:`\delta=0` and perform inference only over the parameters
    :math:`\boldsymbol\beta` and :math:`v`, which means that
    :math:`\mathrm K = v \mathrm Q_0 \mathrm S_0 \mathrm Q_0^\intercal`.

    Args:
        success (array_like): Array of :math:`y_i \in \{0, 1\}`.
        M (array_like): :math:`\mathrm M` covariates.
        Q0 (array_like): :math:`\mathrm Q_0` of the eigendecomposition.
        Q1 (array_like): :math:`\mathrm Q_1` of the eigendecomposition.
        S0 (array_like): :math:`\mathrm S_0` of the eigendecomposition.
        Q0S0Q0t (array_like): :math:`\mathrm Q_0 \mathrm S_0
                        \mathrm Q_0^{\intercal}` in case this has already
                        been computed. Defaults to `None`.
    """

    def __init__(self, success, M, Q0, Q1, S0, Q0S0Q0t=None):
        super(BernoulliEP, self).__init__(M, Q0, S0, False, QSQt=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        success = asarray(success, float)
        self._success = success

        if issingleton(success):
            msg = "The phenotype array has a single unique value only."
            raise ValueError(msg)

        if not all(isfinite(success)):
            raise ValueError("There are non-finite numbers in phenotype.")

        msg = 'Number of individuals mismatch.'
        assert success.shape[0] == M.shape[0], msg
        assert success.shape[0] == Q0.shape[0], msg
        assert success.shape[0] == Q1.shape[0], msg

        self._Q1 = Q1

        self._moments = LikNormMoments(350)
        self.initialize()

    @property
    def genetic_variance(self):
        r"""Returns :math:`\sigma_b^2`."""
        return self.sigma2_b

    @property
    def environmental_variance(self):
        r"""Returns :math:`\pi^2/3`."""
        return (pi * pi) / 3

    @property
    def heritability(self):
        r"""Returns :math:`\sigma_b^2/(\sigma_a^2+\sigma_b^2+\pi^2/3)`."""
        total = self.genetic_variance + self.covariates_variance
        total += self.environmental_variance
        return self.genetic_variance / total

    def initialize(self):
        y = self._success
        ratio = sum(y) / float(len(y))
        latent_mean = st.norm(0, 1).isf(1 - ratio)
        latent = y / y.std()
        latent = latent - latent.mean() + latent_mean

        Q0 = self._Q
        Q1 = self._Q1
        S0 = self._S
        covariates = self._M
        flmm = FastLMM(full(len(y), latent), covariates, QS=((Q0, Q1), (S0,)))
        flmm.learn()
        gv = flmm.genetic_variance
        nv = flmm.environmental_variance
        h2 = gv / (gv + nv)
        h2 = _h2_correction(h2, ratio, ratio)
        h2 = clip(h2, 0.01, 0.9)

        mean = flmm.mean
        self._tbeta = lstsq(self._tM, full(len(y), mean))[0]
        self.delta = 0.
        self.v = self.environmental_variance * (h2 / (1 - h2))

    def _tilted_params(self):
        y = self._success
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz
        self._moments.binomial(y, ones(len(y)), ceta,
                               ctau, lmom0, self._hmu, self._hvar)


def _h2_correction(h2, prevalence, ascertainment):
    t = st.norm.ppf(1 - prevalence)
    z = st.norm.pdf(t)
    k = prevalence * (1 - prevalence)
    p = ascertainment * (1 - ascertainment)
    return h2 * k**2 / (z**2 * p)
