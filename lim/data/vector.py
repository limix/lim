from numpy import asarray
from numpy import atleast_1d

from bidict import bidict

class Vector(object):
    def __init__(self, labels, values):
        self._map = bidict(zip(labels, values))
        self._data = asarray(values)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, args):
        idx = atleast_1d(args)
        return asarray([self._map[i] for i in idx])
