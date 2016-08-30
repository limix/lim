class NormalModel(object):

    def __init__(self, covariate_effect_sizes, fixed_effects_variance,
                 heritability, genetic_variance, environmental_variance,
                 total_variance):
        self.covariate_effect_sizes = covariate_effect_sizes
        self.fixed_effects_variance = fixed_effects_variance
        self.heritability = heritability
        self.genetic_variance = genetic_variance
        self.environmental_variance = environmental_variance
        self.total_variance = total_variance

    def __str__(self):
        d = dict()
        d['ces'] = self.covariate_effect_sizes
        d['fev'] = self.fixed_effects_variance
        d['her'] = self.heritability
        d['gva'] = self.genetic_variance
        d['eva'] = self.environmental_variance
        d['tva'] = self.total_variance
        d = {k: "%7.4f" % v for (k, v) in d.items()}
        s = """Model statistics:
  Covariate effect sizes: {ces}
  Fixed-effect variances: {fev}
  Heritability:           {her}
  Genetic variance:       {gva}
  Environmental variance: {eva}
  Total variance:         {tva}"""
        return s.format(**d)
