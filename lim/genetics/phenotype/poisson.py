from ._phenotype import Phenotype


class PoissonPhenotype(Phenotype):
    def __init__(self, noccurrences):
        super(PoissonPhenotype, self).__init__('Poisson')
        self.noccurrences = noccurrences
