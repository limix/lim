from ._phenotype import Phenotype


class BinomialPhenotype(Phenotype):
    def __init__(self, nsuccesses, ntrials):
        super(BinomialPhenotype, self).__init__('Binomial')
        self.nsuccesses = nsuccesses
        self.ntrials = ntrials
