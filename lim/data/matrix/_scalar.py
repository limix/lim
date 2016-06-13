# from numpy import float64, int64, bytes_
#
# class FloatView(float64):
#     def __new__(cls, *_):
#         return super(FloatView, cls).__new__(cls)
#
#     def __init__(self, ref, lslice):
#         self._ref = ref
#         self._lslice = lslice
#
#     def __eq__(self, rhs):
#         return self.item() == rhs
#
#     def item(self, *args):
#         if len(args) == 0:
#             args = (0,)
#         return self._ref._item(args, [self._lslice])
#
#     @property
#     def raw(self):
#         return float(self.item((0,)))
#
#     @property
#     def shape(self):
#         return tuple()
#
#     @property
#     def ndim(self):
#         return 0
#
#     @property
#     def dtype(self):
#         return type(self.item())
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __repr__(self):
#         return self.item().__str__()
#
# class IntView(int64):
#     def __new__(cls, *_):
#         return super(IntView, cls).__new__(cls)
#
#     def __init__(self, ref, lslice):
#         self._ref = ref
#         self._lslice = lslice
#
#     def __eq__(self, rhs):
#         return self.item() == rhs
#
#     def item(self, *args):
#         if len(args) == 0:
#             args = (0,)
#         return self._ref._item(args, [self._lslice])
#
#     @property
#     def raw(self):
#         return int(self.item((0,)))
#
#     @property
#     def shape(self):
#         return tuple()
#
#     @property
#     def ndim(self):
#         return 0
#
#     @property
#     def dtype(self):
#         return type(self.item())
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __repr__(self):
#         return self.item().__str__()
#
# class BytesView(bytes_):
#     def __new__(cls, *_):
#         return super(BytesView, cls).__new__(cls)
#
#     def __init__(self, ref, lslice):
#         super(BytesView, self).__init__()
#         self._ref = ref
#         self._lslice = lslice
#
#     def __eq__(self, rhs):
#         return self.item() == rhs
#
#     def item(self, *args):
#         if len(args) == 0:
#             args = (0,)
#         return self._ref._item(args, [self._lslice])
#
#     @property
#     def raw(self):
#         return bytes(self.item((0,)))
#
#     @property
#     def shape(self):
#         return tuple()
#
#     @property
#     def ndim(self):
#         return 0
#
#     @property
#     def dtype(self):
#         return type(self.item())
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __repr__(self):
#         return self.item().__str__()
