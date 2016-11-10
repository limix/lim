from numpy import ascontiguousarray

class BernoulliPhenotype(object):
    def __init__(self, outcome):
        self.outcome = ascontiguousarray(outcome, dtype=float)
        self.likelihood_name = 'Bernoulli'

    @property
    def sample_size(self):
        return len(self.outcome)

    def to_normal(self):
        y = self.outcome / self.outcome.std()
        y -= y.mean()
        return y
