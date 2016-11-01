from __future__ import absolute_import
from __future__ import unicode_literals


import logging

from hcache import Cached, cached

from numpy import asarray
from numpy import empty
from numpy import nan

from progressbar import ProgressBar
from progressbar import NullBar
from progressbar import Counter
from progressbar import AdaptiveETA
from progressbar import UnknownLength

from ...util import unicode_compatible


def _indent(msg):
    return '\n'.join(['    ' + s for s in msg.split('\n')])


@unicode_compatible
class QTLScan(Cached):
    def __init__(self, X, progress=True):
        Cached.__init__(self)
        self._logger = logging.getLogger(__name__)

        self._X = X
        self._null_lml = nan
        self._alt_lmls = empty(X.shape[1])
        self._effect_sizes = empty(X.shape[1])
        self.progress = progress

    @property
    def candidate_markers(self):
        """Candidate markers.

        :getter: Returns candidate markers
        :setter: Sets candidate markers
        :type: `array_like` (:math:`N\\times P_c`)
        """

        return self._X

    @candidate_markers.setter
    def candidate_markers(self, X):
        self._X = X
        self.clear_cache('_compute_alt_models')

    def compute_statistics(self):
        self._logger.info('Computing Likelihood-ratio test statistics.')

        self._logger.info('Null model computation has started.')
        if self.progress:
            msg = "Null model fitting: "
            widgets = [msg, Counter(), " function evaluations"]
            progress = ProgressBar(widgets=widgets, max_value=UnknownLength)
        else:
            progress = NullBar()
        self._compute_null_model(progress)
        self._logger.info('Null model computation has finished.')

        self._logger.info('Alternative models computation have started.')
        if self.progress:
            msg = "Scanning markers: "
            widgets = [msg, AdaptiveETA()]
            progress = ProgressBar(widgets=widgets, max_value=self._X.shape[1])
        else:
            progress = NullBar()
        self._compute_alt_models(progress)
        self._logger.info('Alternative models computation have finished.')

    @cached
    def _compute_null_model(self, progress):
        raise NotImplementedError

    @cached
    def _compute_alt_models(self, progress):
        raise NotImplementedError

    def null_lml(self):
        """Log marginal likelihood for the null hypothesis."""
        self._compute_statistics()
        return self._null_lml

    def alt_lmls(self):
        """Log marginal likelihoods for the alternative hypothesis."""
        self._compute_statistics()
        return self._alt_lmls

    def candidate_effect_sizes(self):
        """Effect size for candidate markers."""
        self._compute_statistics()
        return self._candidate_effect_sizes

    def pvalues(self):
        """Association p-value for candidate markers."""
        self._compute_statistics()

        lml_alts = self.alt_lmls()
        lml_null = self.null_lml()

        lrs = -2 * lml_null + 2 * asarray(lml_alts)

        from scipy.stats import chi2
        chi2 = chi2(df=1)

        return chi2.sf(lrs)

    def null_model(self):
        """Model of the null hypothesis."""
        raise NotImplementedError

    def alt_models(self):
        """Model of the alternative hypotheses."""
        raise NotImplementedError

        # snull = str(self.null_model())
        # snull = 'Null model:\n\n' + _indent(snull)
        #
        # salt = self.alt_models()
        # salt = 'Alternative model:\n\n' + _indent(salt)
        #
        # sces = 'Candidate effect sizes:\n'
        # sces += _indent(quantile_summary(self._candidate_effect_sizes))
        # sces = _indent(sces)
        #
        # salmls = 'Candidate log marginal likelihoods:\n'
        # salmls += _indent(quantile_summary(self._alt_lmls))
        # salmls = _indent(salmls)
        #
        # spval = 'Candidate p-values:\n'
        # spval += _indent(quantile_summary(self.pvalues(), "e"))
        # spval = _indent(spval)
        #
        # return '\n\n'.join([snull, salt, sces, salmls, spval])
