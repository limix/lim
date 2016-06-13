# def check_lslice_boundary(lslice, shape):
#     for (i, s) in enumerate(lslice):
#         if isinstance(s, int):
#             if s >= shape[i] or s < 0:
#                 raise IndexError
#         else:
#             if s.stop is not None and s.stop > shape[i]:
#                 raise IndexError
#
# def check_index(i, slice_):
#     if slice_.stop is not None and i * slice_.step >= slice_.stop:
#         return -1
#     if i >= slice_.start and (i-slice_.start) % slice_.step == 0:
#         return 1
#     return 0
#
# def normalize_lslice_shape(lslice, shape):
#     nlslice = []
#     for (i, s) in enumerate(lslice):
#         if isinstance(s, int):
#             nlslice.append(s)
#             continue
#         start = s.start
#         stop = s.stop
#         step = s.step
#
#         if start is None:
#             start = 0
#         elif start < 0:
#             start = shape[i] + start
#
#         if stop is None:
#             stop = shape[i]
#         elif stop < 0:
#             stop = shape[i] + stop
#
#         nlslice.append(slice(start, stop, step))
#     return tuple(nlslice)
#
# def normalize_lslice_dim(lslice, ndim):
#     return lslice + (slice(0, None, 1),) * (ndim - len(lslice))
#
# def _normalize_slice(s):
#     if isinstance(s, int):
#         return s
#     if s.start is None:
#         s = slice(0, s.stop, s.step)
#     if s.step is None:
#         s = slice(s.start, s.stop, 1)
#     return s
#
# def create_lslice(args):
#     return tuple([_normalize_slice(a) for a in args])
#
# def _merge_2lslices(a, b):
#     a = _normalize_slice(a)
#     b = _normalize_slice(b)
#
#     if isinstance(b, int):
#         if b >= 0:
#             return a.start + b*a.step
#         if a.stop is None:
#             return b*a.step
#         return a.stop + b*a.step
#
#     start = a.start + b.start*a.step
#
#     stop = None
#     if b.stop is not None:
#         stop = (b.stop - b.start)*a.step
#
#     step = a.step*b.step
#
#     return slice(start, stop, step)
#
# def _recurse(head, tail):
#     if len(tail) == 1:
#         return _merge_2lslices(head, tail[0])
#     return _merge_2lslices(head, _recurse(tail[0], tail[1:]))
#
# def _merge_slices(slices):
#     if len(slices) == 1:
#         return _normalize_slice(slices[0])
#     slices = list(slices)
#     return _recurse(slices[0], slices[1:])
#
# def merge_lslices(lslices):
#     if len(lslices) == 0:
#         return tuple()
#     n = len(lslices)
#     ndim = len(lslices[0])
#     lslice = []
#     for i in range(ndim):
#         slices = [lslices[j][i] for j in range(n)]
#         lslice.append(_merge_slices(slices))
#     return tuple(lslice)
#
# def extract_ndim(lslice):
#     return sum(isinstance(s, slice) for s in lslice)
#
# def transform_index(idx, lslice):
#     ndim = len(lslice)
#
#     nidx = []
#     j = 0
#     for i in range(ndim):
#         s = lslice[i]
#         if isinstance(s, slice):
#             nidx.append(s.start + idx[j] * s.step)
#             j += 1
#         else:
#             nidx.append(s)
#
#     return tuple(nidx)
#
# def transform_shape(shape, slices):
#     if slices is None:
#         return shape
#
#     try:
#         n = len(slices)
#         slices = [_normalize_slice(s) for s in slices]
#     except TypeError:
#         slices = [_normalize_slice(slices)]
#         n = 1
#
#     new_shapes = []
#     for i in range(n):
#         sh = shape[i]
#         sl = slices[i]
#         if sl.stop is not None:
#             if sl.stop < 0:
#                 sh += sl.stop
#             else:
#                 sh = sl.stop
#         sh -= sl.start
#         new_shapes.append(sh // sl.step)
#
#     return tuple(new_shapes)
