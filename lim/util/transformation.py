from __future__ import division

from numpy import sqrt

class DesignMatrixTrans(object):
    def __init__(self, G):
        self._sub = G.mean(0)
        self._div = G.std(0)
        self._div[self._div == 0] = 1.0
        self._div *= sqrt(G.shape[1])

    def transform(self, X):
        return (X - self._sub) / self._div
