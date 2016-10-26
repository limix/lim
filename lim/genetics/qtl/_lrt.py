from __future__ import absolute_import
from __future__ import unicode_literals

import sys

import logging

from numpy import concatenate
from numpy import asarray
from numpy import newaxis
from numpy import hstack
from numpy import array
from numpy import ones

from progressbar import ProgressBar
from progressbar import NullBar
from progressbar import Percentage
from progressbar import UnknownLength
from progressbar import Counter
from progressbar import AdaptiveETA

from ...util import quantile_summary
from ...util import unicode_compatible


def _indent(msg):
    return '\n'.join(['    ' + s for s in msg.split('\n')])


@unicode_compatible
class LikelihoodRatioTestScan(object):

    def __init__(self, progress=True):

        self._logger = logging.getLogger(__name__)

        self._progress = progress

        self._X = None
        self._null_lml = None
        self._alt_lmls = None
        self._candidate_effect_sizes = None
        self._null_model_ready = False
        self._alt_model_ready = False

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
        self._alt_model_ready = False

    def _compute_statistics(self):
        self._logger.info('Computing Likelihood-ratio test statistics.')
        self._compute_null_model()
        self._compute_alt_models()

    def _compute_null_model(self):
        if self._null_model_ready:
            return
        self._logger.info('Null model computation has started.')

        if self._progress:
            msg = "Null model fitting: "
            progress = ProgressBar(widgets=[msg, Counter(),
                                            " function evaluations"],
                                   max_value=UnknownLength)
        else:
            progress = NullBar()

        self._learn_null_model(progress)

        self._null_model_ready = True

    def _compute_alt_models(self):
        if self._alt_model_ready:
            return
        self._logger.info('Alternative model computation has started.')

        nmarkers = self._X.shape[1]

        if self._progress:
            msg = "Scanning markers: "
            progress = ProgressBar(widgets=[msg, AdaptiveETA()],
                                   max_value=nmarkers)
        else:
            progress = NullBar()

        self._learn_alt_models(progress)

        self._alt_model_ready = True

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
        raise NotImplementError

    def alt_models(self):
        """Model of the alternative hypotheses."""
        raise NotImplementError

    def _learn_null_model(self, progress):
        raise NotImplementError

    def _learn_alt_models(self, progress):
        raise NotImplementError

    def __str__(self):
        return ""
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
