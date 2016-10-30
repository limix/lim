from ._phenotype import Phenotype


class NormalPhenotype(Phenotype):
    def __init__(self, y):
        super(NormalPhenotype, self).__init__('Normal')
        self._y = y

    def __str__(self):
        lik = self._likelihood_name
        t = [['Likelihood', lik[0].upper() + lik[1:]]]
        if self.likelihood == 'normal':
            t.append(['Phenotype mean', self.phenotype.mean()])
            t.append(['Phenotype std', self.phenotype.std()])
            mima = (self.phenotype.min(), self.phenotype.max())
            t.append(['Phenotype (min, max)', mima])
        return t
