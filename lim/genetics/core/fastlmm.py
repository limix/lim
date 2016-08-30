from __future__ import division

from numpy import exp
from numpy import clip
from numpy import zeros
from numpy import isfinite
from numpy import atleast_2d
from numpy import all as all_
from numpy import set_printoptions

from limix_math.linalg import qs_decomposition

from ..transformation import DesignMatrixTrans
from ..model import NormalModel
from ...inference import FastLMM as FastLMMCore
from ...func import maximize_scalar
from ...func import Learnable
from ...func import Variables
from ...func import Scalar
from ...func import FuncData


class FastLMM(Learnable, FuncData):

    def __init__(self, y, covariates, X=None, QS=None):
        self._logistic = Scalar(0.0)
        Learnable.__init__(self, Variables(logistic=self._logistic))
        FuncData.__init__(self)

        assert (X is None) != (QS is None)
        if not all_(isfinite(y)):
            raise ValueError("There are non-finite values in the phenotype.")

        self._trans = None
        if QS is None:
            self._trans = DesignMatrixTrans(X)
            X = self._trans.transform(X)
            QS = qs_decomposition(X)
            self._X = X

        self._flmmc = FastLMMCore(y, covariates, QS[0][0], QS[0][1], QS[1][0])

    @property
    def covariates(self):
        return self._flmmc.covariates

    @covariates.setter
    def covariates(self, v):
        self._flmmc.covariates = v

    def copy(self):
        o = FastLMM.__new__(FastLMM)
        o._logistic = self._logistic.copy()
        Learnable.__init__(o, Variables(logistic=o._logistic))
        FuncData.__init__(o)
        o._flmmc = self._flmmc.copy()
        o._trans = self._trans
        return o

    def _delta(self):
        v = clip(self._logistic.value, -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, 1e-5, 1 - 1e-5)

    @property
    def heritability(self):
        t = (self.fixed_effects_variance + self.genetic_variance
             + self.environmental_variance)
        return self.genetic_variance / t

    @property
    def fixed_effects_variance(self):
        return self._flmmc.mean.var()

    @property
    def genetic_variance(self):
        return self._flmmc.scale * (1 - self._flmmc.delta)

    @property
    def environmental_variance(self):
        return self._flmmc.scale * self._flmmc.delta

    @property
    def beta(self):
        return self._flmmc.beta

    @property
    def mean(self):
        return self._flmmc.mean

    def learn(self, progress=None):
        maximize_scalar(self, progress)
        self._flmmc.delta = self._delta()

    def value(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()

    def lml(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()

    def predict(self, covariates, Xp):
        covariates = atleast_2d(covariates)
        Xp = atleast_2d(Xp)
        Xp = self._trans.transform(Xp)
        Cp = Xp.dot(self._X.T)
        Cpp = Xp.dot(Xp.T)
        return self._flmmc.predict(covariates, Cp, Cpp)

    def model(self):
        total_var = (self.fixed_effects_variance + self.genetic_variance
                     + self.environmental_variance)
        return NormalModel(self.beta, self.fixed_effects_variance,
                           self.heritability, self.genetic_variance,
                           self.environmental_variance, total_var)

    def __str__(self):
        v = self.genetic_variance
        e = self.environmental_variance
        beta = self.beta
        cvar = self.fixed_effects_variance
        tvar = cvar + v + e
        h2 = self.heritability

        var_sym = unichr(0x3bd).encode('utf-8')
        set_printoptions(precision=3, threshold=10)
        s = """Phenotype:
  y_i = o_i + u_i + e_i

Definitions:
  M: covariates
  o: fixed-effects signal = M {b}.T
  u: background signal    ~ Normal(0, {v} * Kinship)
  e: environmental signal ~ Normal(0, {e} * I)

Log marginal likelihood: {lml}

Statistics (latent space):
  Total variance:         {tvar}     {vs}_o + {vs}_u + {vs}_e
  Fixed-effect variances: {cvar}     {vs}_o
  Heritability:           {h2}     {vs}_u / ({vs}_o + {vs}_u + {vs}_e)
where {vs}_x is the variance of signal x"""\
        .format(v="%7.4f" % v, e="%7.4f" % e, b=beta,
                tvar="%7.4f" % tvar, cvar="%7.4f" % cvar, h2="%7.4f" % h2,
                vs=var_sym, lml="%9.6f" % self.lml())
        set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                         precision=8, suppress=False, threshold=1000,
                         formatter=None)
        return s
