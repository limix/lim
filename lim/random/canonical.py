from numpy import sqrt
from numpy import std
from limix_inference.link import LogitLink
from limix_inference.link import LogLink
from limix_inference.lik import BernoulliLik
from limix_inference.lik import BernoulliProdLik
from limix_inference.lik import BinomialProdLik
from limix_inference.lik import PoissonProdLik
from limix_inference.mean import OffsetMean
from limix_inference.mean import LinearMean
from limix_inference.mean import SumMean
from limix_inference.cov import LinearCov
from limix_inference.cov import SumCov
from limix_inference.cov import EyeCov
from .glmm import GLMMSampler
from ..util.fruits import Apples
from ..tool.normalize import stdnorm


def bernoulli(offset, G, heritability=0.5, causal_variants=None,
              causal_variance=0, random_state=None):

    link = LogitLink()
    mean, cov = _mean_cov(offset, G, heritability, causal_variants,
                          causal_variance, random_state)
    lik = BernoulliProdLik(link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def binomial(ntrials,
             offset,
             G,
             heritability=0.5,
             causal_variants=None,
             causal_variance=0,
             random_state=None):

    link = LogitLink()
    mean, cov = _mean_cov(offset, G, heritability, causal_variants,
                          causal_variance, random_state)
    lik = BinomialProdLik(ntrials, link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def poisson(offset, G, heritability=0.5, causal_variants=None,
            causal_variance=0, random_state=None):

    mean, cov = _mean_cov(offset, G, heritability, causal_variants,
                          causal_variance, random_state)
    link = LogLink()
    lik = PoissonProdLik(link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def _causal_mean(causal_variants, causal_variance, random):
    causal_variants = stdnorm(causal_variants, axis=0)
    causal_variants /= sqrt(causal_variants.shape[1])
    p = causal_variants.shape[1]
    directions = random.randn(p)
    directions[directions < 0.5] = -1
    directions[directions >= 0.5] = +1
    s = std(directions)
    if s > 0:
        directions /= s
    directions *= sqrt(causal_variance)
    directions -= directions.mean()
    mean = LinearMean(p)
    mean.set_data((causal_variants, ), 'sample')
    mean.effsizes = directions
    return mean

def _mean_cov(offset, G, heritability, causal_variants, causal_variance,
              random_state):
    nsamples = G.shape[0]
    G = stdnorm(G, axis=0)

    G /= sqrt(G.shape[1])

    mean1 = OffsetMean()
    mean1.offset = offset

    cov1 = LinearCov()
    cov2 = EyeCov()
    cov = SumCov([cov1, cov2])

    mean1.set_data(nsamples, 'sample')
    cov1.set_data((G, G), 'sample')
    a = Apples(nsamples)
    cov2.set_data((a, a), 'sample')

    cov1.scale = heritability - causal_variance
    cov2.scale = 1 - heritability - causal_variance

    means = [mean1]
    if causal_variants is not None:
        means += [_causal_mean(causal_variants, causal_variance, random_state)]

    mean = SumMean(means)

    return mean, cov
