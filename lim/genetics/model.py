from __future__ import unicode_literals

from six import string_types

from numpy import ndarray

from ..util import get_greek
from ..util import unicode_compatible


def _stringit(v):
    if isinstance(v, ndarray):
        return '[' + (', '.join(["%7.4f" % vi for vi in v])) + ']'
    elif isinstance(v, string_types):
        return v
    return "%7.4f" % v


@unicode_compatible
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

        d = {k: _stringit(v) for (k, v) in d.items()}

        s = """Phenotype:
    y_i = o_i + u_i + e_i

Definitions:
    M: covariates
    o: fixed-effects signal = M {ces}.T
    u: background signal    ~ Normal(0,  {gva} * Kinship)
    e: environmental signal ~ Normal(0,  {eva} * I)"""

        s += "\n\n"

        s += """Model statistics:
    Covariate effect sizes: {ces}
    Fixed-effect variances: {fev}
    Heritability:           {her}
    Genetic variance:       {gva}
    Environmental variance: {eva}
    Total variance:         {tva}"""

        return s.format(**d)


@unicode_compatible
class BinomialModel(object):

    def __init__(self, covariate_effect_sizes, fixed_effects_variance,
                 real_variance, noise_ratio, genetic_variance,
                 environmental_variance, instrumental_variance,
                 environmental_genetic_ratio, genetic_ratio, heritability):

        self.covariate_effect_sizes = covariate_effect_sizes
        self.fixed_effects_variance = fixed_effects_variance
        self.real_variance = real_variance
        self.noise_ratio = noise_ratio
        self.genetic_variance = genetic_variance
        self.environmental_variance = environmental_variance
        self.instrumental_variance = instrumental_variance
        self.environmental_genetic_ratio = environmental_genetic_ratio
        self.genetic_ratio = genetic_ratio
        self.heritability = heritability

    def __str__(self):
        d = dict()
        d['ces'] = self.covariate_effect_sizes
        d['fev'] = self.fixed_effects_variance

        d['nra'] = self.noise_ratio
        d['gva'] = self.genetic_variance
        d['eva'] = self.environmental_variance
        d['iva'] = self.instrumental_variance
        d['egr'] = self.environmental_genetic_ratio
        d['gra'] = self.genetic_ratio
        d['her'] = self.heritability

        d['eps'] = get_greek('epsilon')
        d['sum'] = get_greek('Sigma')
        d['bone'] = u'\U0001d7cf'

        d = {k: _stringit(v) for (k, v) in d.items()}

        s = """Phenotype:
    y_i = {sum}_{{j=1}}^{{n_i}} {bone}(f_i + {eps}_{{i,j}} > 0)

Latent phenotype:
    f_i = o_i + u_i + e_i

Definitions:
    M      : covariates
    o      : fixed-effects signal = M {ces}.T
    u      : background signal    ~ Normal(0,  {gva} * Kinship)
    e      : environmental signal ~ Normal(0,  {eva} * I)
    {eps}_{{i,j}}: instrumental signal  ~ Normal(0,   1)
    n_i    : number of draws of the i-th sample"""

        s += "\n\n"

        s += """Model statistics:
    Covariate effect sizes: {ces}
    Fixed-effect variances: {fev}
    Genetic variance:       {gva}
    Environmental variance: {eva}
    Instrumental variance:  {iva}
    Heritability:           {her}
    Genetic ratio:          {gra}
    Noise ratio:            {nra}"""

        return s.format(**d)
