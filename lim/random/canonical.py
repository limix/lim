from numpy import sqrt
from ..link import LogitLink
from ..lik import BernoulliLik
from ..lik import BernoulliProdLik
from ..lik import BinomialProdLik
from ..mean import OffsetMean
from ..cov import LinearCov
from ..cov import SumCov
from ..cov import EyeCov
from .glmm import GLMMSampler
from ..util.fruits import Apples
from ..tool.normalize import stdnorm


def bernoulli(offset, G, heritability=0.5, random_state=None):

    nsamples = G.shape[0]
    G = stdnorm(G, axis=0)
    G /= sqrt(G.shape[1])

    link = LogitLink()

    mean = OffsetMean()
    mean.offset = offset

    cov = LinearCov()

    mean.set_data(nsamples, 'sample')
    cov.set_data((G, G), 'sample')

    r = heritability / (1 - heritability)
    cov.scale = BernoulliLik.latent_variance(link) * r

    lik = BernoulliProdLik(None, link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def binomial(ntrials, offset, G, heritability=0.5, random_state=None):

    nsamples = G.shape[0]
    G = stdnorm(G, axis=0)
    G /= sqrt(G.shape[1])

    link = LogitLink()

    mean = OffsetMean()
    mean.offset = offset

    cov1 = LinearCov()
    cov2 = EyeCov()
    cov = SumCov([cov1, cov2])

    mean.set_data(nsamples, 'sample')
    cov1.set_data((G, G), 'sample')
    a = Apples(nsamples)
    cov2.set_data((a, a), 'sample')

    cov1.scale = heritability
    cov2.scale = 1 - heritability

    lik = BinomialProdLik(None, ntrials, link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)
