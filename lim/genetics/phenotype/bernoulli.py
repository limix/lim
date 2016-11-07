from ._phenotype import Phenotype


class BernoulliPhenotype(Phenotype):
    def __init__(self, outcome):
        super(BernoulliPhenotype, self).__init__('Bernoulli')
        self.outcome = outcome
