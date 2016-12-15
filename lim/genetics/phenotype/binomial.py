from __future__ import absolute_import, division, unicode_literals

from numpy import ascontiguousarray

from numpy_sugar import (is_all_equal, is_all_finite)
from limix_inference.lik import BinomialProdLik
from limix_inference.link import LogitLink

class BinomialPhenotype(object):
    def __init__(self, nsuccesses, ntrials):
        self.nsuccesses = ascontiguousarray(nsuccesses, dtype=float)
        self.ntrials = ascontiguousarray(ntrials, dtype=float)
        self.likelihood_name = 'Binomial'

        if is_all_equal(nsuccesses):
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

    def to_likelihood(self):
        lik = BinomialProdLik(self.ntrials, LogitLink())
        lik.nsuccesses = self.nsuccesses
        return lik
