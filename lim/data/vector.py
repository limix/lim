from collections import MutableMapping

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

    def items(self):
        return VectorView(self, self._map)

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return bytes(self._data)

class VectorView(MutableMapping):
    def __init__(self, _ref, map_):
        self._ref, self._map = _ref, map_

    def __getitem__(self, key):
        if key in self._map.keys():
            return self._map[key]
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        for (key, val) in iter(self._map.items()):
            yield key, val

    def __setitem__(self, key, value):
        if key in self._map:
            self._map[key] = value
        else:
            raise KeyError(key)

    def __delitem__(self, key):
        self._map.remove(key)
