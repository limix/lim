# from numpy import array
#
# class ArrayViewBase(object):
#     def __init__(self):
#         pass
#
#     def set_axis_name(self, axis, name):
#         raise NotImplementedError
#
#     def get_axis_name(self, axis):
#         raise NotImplementedError
#
#     def set_axis_values(self, axis, values):
#         raise NotImplementedError
#
#     def get_axis_values(self, axis):
#         raise NotImplementedError
#
#     def item(self, *args, **kwargs):
#         raise NotImplementedError
#
#     def __getitem__(self, slice_):
#         raise NotImplementedError
#
#     @property
#     def shape(self):
#         raise NotImplementedError
#
#     def __iter__(self):
#         for i in range(self.shape[0]):
#             yield self[i].raw
#
#     def __len__(self):
#         return self.shape[0]
#
#     @property
#     def ndim(self):
#         raise NotImplementedError
#
#     @property
#     def dtype(self):
#         raise NotImplementedError
#
#     def __repr__(self):
#         return bytes(self._create_array_recurse([]))
#
#     def __str__(self):
#         return repr(self)
#
#     def _create_array(self, lslice):
#         raise NotImplementedError
#
#     def _create_array_recurse(self, lslice_list):
#         raise NotImplementedError
#
#     def __array__(self):
#         return self._create_array_recurse([])
#
#     def __eq__(self, rhs):
#         lia = self.get_axis_values(self.indexed_axis)
#         ria = self.get_axis_values(rhs.indexed_axis)
#
#         lok = array([ia in set(ria) for ia in lia])
#         rok = array([ia in set(lia) for ia in ria])
#
#         larr = self.__array__()[lok]
#         rarr = rhs.__array__()[rok]
#
#         return ArrayIdx(lia[lok][larr == rarr])
#
#         return self.__array__() == rhs
#
#     def __ge__(self, rhs):
#         return self.__array__() >= rhs
#
#     def __le__(self, rhs):
#         return self.__array__() <= rhs
#
#     def __gt__(self, rhs):
#         return self.__array__() > rhs
#
#     def __lt__(self, rhs):
#         return self.__array__() < rhs
#
#     def __ne__(self, rhs):
#         return self.__array__() != rhs
