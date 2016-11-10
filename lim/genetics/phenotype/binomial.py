from __future__ import absolute_import, division, unicode_literals

from numpy import ascontiguousarray

from limix_math import (issingleton, is_all_finite)

class BinomialPhenotype(object):
    def __init__(self, nsuccesses, ntrials):
        self.nsuccesses = ascontiguousarray(nsuccesses, dtype=float)
        self.ntrials = ascontiguousarray(ntrials, dtype=float)
        self.likelihood_name = 'Binomial'

        if issingleton(nsuccesses):
            raise ValueError("The phenotype array has a single unique value" +
                             " only.")

        if not is_all_finite(nsuccesses):
            raise ValueError("There are non-finite numbers in phenotype.")

    @property
    def sample_size(self):
        return len(self.nsuccesses)

    def to_normal(self):
        y = self.nsuccesses / self.ntrials
        y = y / y.std()
        y -= y.mean()
        return y
