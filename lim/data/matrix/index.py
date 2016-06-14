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
        return (Slice(args, args+1, 1),
                _from_slice(slice(None, None, None), shape[1]))

    if isinstance(args, slice):
        l = _from_slice(args, shape[0])
        r = _from_slice(slice(None, None, None), shape[1])
        return (l, r)

    if isinstance(args, tuple):
        if len(args) == 1:
            if isinstance(args[0], int):
                return (Slice(args[0], args[0]+1, 1), _from_slice(slice(None, None, None), shape[1]))

            l = _from_slice(args[0], shape[0])
            r = _from_slice(slice(None, None, None), shape[1])
            return (l, r)

        if isinstance(args[0], int):
            l = Slice(args[0], args[0]+1, 1)
        else:
            l = _from_slice(args[0], shape[0])

        if isinstance(args[1], int):
            r = Slice(args[1], args[1]+1, 1)
        else:
            r = _from_slice(args[1], shape[1])

        return (l, r)

    raise ValueError

def _merge_2slices(a, b):
    if isinstance(a, int) and isinstance(b, int):
        raise IndexError("Cannot merge two Slices.")

    if isinstance(b, int):
        if b >= 0:
            return a.start + b*a.step
        return a.stop + b*a.step

    start = a.start + b.start*a.step
    stop = (b.stop - b.start)*a.step
    step = a.step*b.step

    return Slice(start, stop, step)

def _recurse(head, tail):
    if len(tail) == 1:
        return _merge_2slices(head, tail[0])
    return _merge_2slices(head, _recurse(tail[0], tail[1:]))

def _merge_slices(slices):
    if len(slices) == 1:
        return slices[0]
    slices = list(slices)
    return _recurse(slices[0], slices[1:])

def merge_mslices(mslices):
    return tuple([_merge_slices([s[0] for s in mslices]),
                  _merge_slices([s[1] for s in mslices])])
