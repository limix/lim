from numpy import sum as sum_
from numpy import dot
from numpy import vstack
from numpy import nan_to_num
from numpy import hstack
from limix_math import cho_solve
from numpy import errstate
from limix_math import ddot


class FixedEP(object):
    def __init__(self, lml_const, A, C, L, Q, QBiQtCteta, teta, beta_nom):
        self._lml_const = lml_const
        self._QBiQtCteta = QBiQtCteta
        self._teta = teta
        self._A = A
        self._C = C
        self._L = L
        self._Q = Q
        self._beta_nom = beta_nom

    def compute(self, covariates, X):
        assert covariates.shape[1] == 1

        Ms = hstack((covariates, X))

        A = self._A
        L = self._L
        Q = self._Q
        C = self._C

        AMs = ddot(A, Ms, left=True)
        dens = AMs - ddot(A, dot(Q, cho_solve(L, dot(Q.T, AMs))), left=True)
        noms = dot(self._beta_nom, Ms)

        row0 = dot(Ms[:, 0], dens)
        row11 = sum_(Ms[:, 1:] * dens[:, 1:], axis=0)

        betas0 = noms[0] * row11[:] - noms[1:] * row0[1:]
        betas1 = -noms[0] * row0[1:] + noms[1:] * row0[0]

        betas = vstack((betas0, betas1))
        denom = row0[0] * row11[:] - row0[1:]**2

        with errstate(divide='ignore'):
            betas /= denom

        betas = nan_to_num(betas)

        ms = dot(covariates, betas[:1, :]) + X * betas[1, :]
        QBiQtCteta = self._QBiQtCteta
        teta = self._teta

        Am = ddot(A, ms, left=True)
        w4 = dot(C * teta, ms) - dot(QBiQtCteta, Am)
        QBiQtAm = dot(Q, cho_solve(L, dot(Q.T, Am)))
        w5 = -(Am * ms).sum(0) / 2 + (Am * QBiQtAm).sum(0) / 2
        lmls = self._lml_const + w4 + w5

        return (lmls, betas[1, :])
