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


class BinomialEP(EP):
    r"""Binomial EP inference.

    Let :math:`p_i` be the probability of success for the i-th individual and

    .. math::

        p(y_i | p_i) = {N_i \choose K_i} p_i^{K_i} (1-p_i)^{N_i-K_i}

    be the Binomial likelihood, where :math:`N_i` and :math:`K_i` are the
    number of trials and successes observed, respectively. Let
    :math:`y_i = K_i/N_i` such that :math:`\mathrm E[y_i|z_i] = p_i`.
    The marginal likelihood is

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(p_i)=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    where

    .. math::

        g(x) = \log \left(\frac{x}{1-x}\right)

    is the Logit link function.

    Args:
        nsuccesses (array_like): Array of :math:`N_i \in \{1, 2, \dots\}`.
        ntrials (array_like): Array of :math:`K_i \in \{0, 1, \dots, N_i\}`.
        M (array_like): :math:`\mathrm M` covariates.
        Q0 (array_like): :math:`\mathrm Q_0` of the eigendecomposition.
        Q1 (array_like): :math:`\mathrm Q_1` of the eigendecomposition.
        S0 (array_like): :math:`\mathrm S_0` of the eigendecomposition.
        Q0S0Q0t (array_like): :math:`\mathrm Q_0 \mathrm S_0
                        \mathrm Q_0^{\intercal}` in case this has already
                        been computed. Defaults to `None`.
    """

    def __init__(self, nsuccesses, ntrials, M, Q0, Q1, S0,
                 Q0S0Q0t=None):
        super(BinomialEP, self).__init__(M, Q0, S0, True, QSQt=Q0S0Q0t)
        self._logger = logging.getLogger(__name__)

        nsuccesses = asarray(nsuccesses, float)

        if isscalar(ntrials):
            ntrials = full(len(nsuccesses), ntrials, dtype=float)
        else:
            ntrials = asarray(ntrials, float)

        self._nsuccesses = nsuccesses
        self._ntrials = ntrials

        if issingleton(nsuccesses):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not all(isfinite(nsuccesses)):
            raise ValueError("There are non-finite numbers in phenotype.")

        assert nsuccesses.shape[0] == M.shape[
            0], 'Number of individuals mismatch.'
        assert nsuccesses.shape[0] == Q0.shape[
            0], 'Number of individuals mismatch.'
        assert nsuccesses.shape[0] == Q1.shape[
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

        nsuccesses = self._nsuccesses
        ntrials = self._ntrials

        latent = nsuccesses / ntrials
        latent = latent / latent.std()
        latent -= latent.mean()

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
        self._tbeta = lstsq(self._tM, full(len(ntrials), mean))[0]
        self.delta = 1 - h2
        self.v = gv + nv

    def _tilted_params(self):
        nsuccesses = self._nsuccesses
        ntrials = self._ntrials
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz

        self._moments.binomial(nsuccesses, ntrials, ceta,
                               ctau, lmom0, self._hmu, self._hvar)
