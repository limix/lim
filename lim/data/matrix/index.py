# class ArrayIdx(object):
#     def __init__(self, index):
#         self._index = index
#         self._name = None
#
#     @property
#     def name(self):
#         return self._name
#
#     @name.setter
#     def name(self, name):
#         self._name = name
class Slice(object):
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __str__(self):
        return self.__str__()

    def __repr__(self):
        return "Slice(%d, %d, %d)" % (self.start, self.stop, self.step)

def _from_slice(slice_, size):
    if isinstance(slice_, Slice):
        return slice_

    start = None
    if slice_.start is None:
        start = 0
    elif slice_.start >= 0:
        start = slice_.start
    else:
        start = size - slice_.start

    stop = None
    if slice_.stop is None:
        stop = size
    elif slice_.stop >= 0:
        stop = slice_.stop
    else:
        stop = size - slice_.stop

    step = None
    if slice_.step is None:
        step = 1
    else:
        step = slice_.step

    return Slice(start, stop, step)

def normalize_access(args, shape):
    if isinstance(args, int):
        return (args, _from_slice(slice(None, None, None), shape[1]))

    if isinstance(args, slice):
        l = _from_slice(args, shape[0])
        r = _from_slice(slice(None, None, None), shape[1])
        return (l, r)

    if isinstance(args, tuple):
        if len(args) == 1:
            if isinstance(args[0], int):
                return (args[0], _from_slice(slice(None, None, None), shape[1]))

            l = _from_slice(args[0], shape[0])
            r = _from_slice(slice(None, None, None), shape[1])
            return (l, r)

        if isinstance(args[0], int):
            l = args[0]
        else:
            l = _from_slice(args[0], shape[0])

        if isinstance(args[1], int):
            r = args[1]
        else:
            r = _from_slice(args[1], shape[1])

        return (l, r)

    raise ValueError
