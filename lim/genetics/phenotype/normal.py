from ._phenotype import Phenotype


class NormalPhenotype(Phenotype):
    def __init__(self, outcome):
        super(NormalPhenotype, self).__init__('Normal')
        self.outcome = outcome
