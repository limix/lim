from ..link import LogitLink
from ..lik import Binomial
from ..mean import OffsetMean
from ..cov import LinearCov
from ..cov import SumCov
from ..cov import EyeCov
from .glmm import GLMMSampler
from ..util.fruits import Apples

def binomial(ntrials, offset, G, random_state=None):

    nsamples = G.shape[0]

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

    lik = Binomial(3, ntrials, link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)
