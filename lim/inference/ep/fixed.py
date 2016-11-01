from numpy import sum as sum_
from numpy import dot
from numpy import vstack
from numpy import maximum
from numpy import minimum
from numpy import where
from numpy import empty
from numpy import all as all_
from limix_math import cho_solve
from limix_math import epsilon


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

        # self._u = dot(Q0B1Q0t, A2teta)

        # AQA = ddot(A1, ddot(Q0B1Q0t, A1, left=False), left=True)
        # self._beta_den = sum2diag(-AQA, A1)

    def lmls(self, ms):
        QBiQtCteta = self._QBiQtCteta
        teta = self._teta
        A = self._A
        C = self._C
        L = self._L
        Q = self._Q

        lmls = empty(ms.shape[1])
        for i in range(ms.shape[1]):
            Am = A * ms[:,i]
            w4 = dot(ms[:,i] * C, teta) - dot(Am, QBiQtCteta)
            QBiQtAm = dot(Q, cho_solve(L, dot(Q.T, Am)))
            w5 = -dot(Am, ms[:,i]) / 2 + dot(Am, QBiQtAm) / 2
            lmls[i] = self._lml_const + w4 + w5

        return lmls

    # I assume that Ms represents several n-by-2 M,
    # where for each candidate i, M[i,0] is the offset
    # and M[i,1] is the variant.
    # Ms[:,0] will have the offset and Ms[:,1:] will have
    # all the candidate variants.
    def optimal_betas(self, Ms, ncovariates):
        assert ncovariates == 1

        A = self._A
        L = self._L
        Q = self._Q

        # L = self._L()
        # Q = self._Q
        # AM = ddot(self._A(), self._tM, left=True)
        # QBiQtAM = dot(Q, cho_solve(L, dot(Q.T, AM)))
        # return dot(self._tM.T, AM) - dot(AM.T, QBiQtAM)
        # AQA = ddot(A1, ddot(Q0B1Q0t, A1, left=False), left=True)
        # self._beta_den = sum2diag(-AQA, A1)

        dens = empty((Ms.shape[0], Ms.shape[1]))
        for i in range(Ms.shape[1]):
            dens[:,i] = A * Ms[:,i]
            dens[:,i] -= A * dot(Q, cho_solve(L, dot(Q.T, A * Ms[:,i])))

        noms = dot(self._beta_nom, Ms)
        # dens = dot(self._beta_den, Ms)

        row0 = dot(Ms[:, 0], dens)
        row11 = sum_(Ms[:, 1:] * dens[:, 1:], axis=0)

        obetas_0 = noms[0] * row11[:] - noms[1:] * row0[1:]
        obetas_1 = -noms[0] * row0[1:] + noms[1:] * row0[0]

        obetas = vstack((obetas_0, obetas_1))
        denom = row0[0] * row11[:] - row0[1:]**2
        denom[denom >= 0.] = maximum(denom[denom >= 0.], epsilon.small)
        denom[denom < 0.] = minimum(denom[denom < 0.], epsilon.small)
        obetas /= denom

        allzero = where(all_(Ms == 0., 0))[0]
        if len(allzero) > 0:
            obetas[0, allzero - 1] = noms[0] / row0[0]
            assert all(obetas[1, allzero - 1] == 0.)

        return obetas
