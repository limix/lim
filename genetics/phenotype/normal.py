from numpy import ascontiguousarray

class NormalPhenotype(object):
    def __init__(self, outcome):
        self.outcome = ascontiguousarray(outcome, dtype=float)
        self.likelihood_name = 'Normal'

    @property
    def sample_size(self):
        return len(self.outcome)
