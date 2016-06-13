# import h5py as h5
#
# from numpy import asarray
# from numpy import atleast_1d
#
# from ..data.array import MatrixInterface
# from ..data.slice import extract_ndim
# from ..util import npy2py_type
#
# class H5Path(MatrixInterface):
#     def __init__(self, filepath, itempath, dtype=None):
#         super(H5Path, self).__init__()
#         self._filepath = filepath
#         self._itempath = itempath
#         if dtype is None:
#             with h5.File(self._filepath, 'r') as f:
#                 harr = f[self._itempath]
#                 dtype = harr[(0,) * harr.ndim].dtype
#                 self._dtype = npy2py_type(dtype)
#         else:
#             self._dtype = dtype
#
#     def item(self, *args):
#         with h5.File(self._filepath, 'r') as f:
#             return self._dtype(f[self._itempath][args])
#
#     @property
#     def dtype(self):
#         return self._dtype
#
#     @property
#     def shape(self):
#         with h5.File(self._filepath, 'r') as f:
#             s = f[self._itempath].shape
#             if len(s) == 1:
#                 return (s[0], 1)
#             return s
#
#     @property
#     def ndim(self):
#         return 2
#
#     def _create_array(self, lslice):
#         with h5.File(self._filepath, 'r') as f:
#             arr = f[self._itempath]
#             if arr.ndim == 1:
#                 v = arr[lslice[0]].astype(self._dtype)
#                 if extract_ndim(lslice) == 2:
#                     return v.reshape((v.shape[0], 1))
#                 return atleast_1d(v)
#             return arr[lslice].astype(self._dtype)
#
# def reader(filepath, itempath, dtype=None):
#     return H5Path(filepath, itempath, dtype=dtype)
