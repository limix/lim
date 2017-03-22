from numpy import ascontiguousarray
from limix_inference.lik import BernoulliProdLik
from limix_inference.link import LogitLink

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

    def to_likelihood(self):
        lik = BernoulliProdLik(LogitLink())
        lik.outcome = self.outcome
        return lik
