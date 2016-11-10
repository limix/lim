from numpy import ascontiguousarray

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
