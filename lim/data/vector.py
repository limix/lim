from collections import MutableMapping

from numpy import asarray
from numpy import atleast_1d

from ..util import npy2py_cast
from ..util import npy2py_type

class Vector(object):
    def __init__(self, labels, values):
        cvalues = [npy2py_cast(v) for v in values]
        self._map = dict(zip(labels, cvalues))

        self._imap = dict()
        for i in range(len(cvalues)):
            if cvalues[i] not in self._imap:
                self._imap[cvalues[i]] = []
            self._imap[cvalues[i]].append(labels[i])

        self._data = asarray(values)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, args):
        if npy2py_type(type(args)) in [int, bytes, float]:
            return npy2py_cast(self._map[args])
        idx = atleast_1d(args)
        return asarray([self._map[i] for i in idx])

    def items(self):
        return VectorView(self, self._map)

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return bytes(self._data)

    @property
    def dtype(self):
        return npy2py_type(self._data.dtype)

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
