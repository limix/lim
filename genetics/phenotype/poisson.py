from numpy import ascontiguousarray

from limix_inference.lik import PoissonProdLik
from limix_inference.link import LogLink

class PoissonPhenotype(object):
    def __init__(self, noccurrences):
        self.noccurrences = ascontiguousarray(noccurrences, dtype=float)
        self.likelihood_name = 'Poisson'

    @property
    def sample_size(self):
        return len(self.noccurrences)

    def to_normal(self):
        y = self.noccurrences
        return (y - y.mean()) / y.std()

    def to_likelihood(self):
        lik = PoissonProdLik(LogLink())
        lik.noccurrences = self.noccurrences
        return lik
