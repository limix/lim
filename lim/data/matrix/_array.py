from collections import Iterable

from numpy import asarray
from numpy import full
from numpy import where

from .scalar import FloatView
from .scalar import BytesView
from .scalar import IntView

from .slice import create_lslice
from .slice import merge_lslices
from .slice import transform_index
from .slice import transform_shape
from .slice import extract_ndim
from .slice import normalize_lslice_dim
from .slice import normalize_lslice_shape
from .slice import check_lslice_boundary
from .base import ArrayViewBase

class ArrayAccessor(object):
    def __init__(self, ref, axis):
        self._ref = ref
        self._axis = axis

    def __call__(self, args):
        return self._ref.get(self._axis, args)

class MatrixInterface(ArrayViewBase):
    def __init__(self):
        super(MatrixInterface, self).__init__()
        self._axis_name = dict()
        self._axis_values = dict()
        self._indexed_axis = None

    @property
    def indexed_axis(self):
        return self._indexed_axis

    @indexed_axis.setter
    def indexed_axis(self, axis):
        self._indexed_axis = axis

    def get(self, axis, label):
        labels = self.get_axis_values(axis)
        if axis == 0:
            return self[where(labels == label)[0][0],:]
        elif axis == 1:
            return self[:,where(labels == label)[0][0]]
        raise IndexError

    def set_axis_name(self, axis, name):
        self._axis_name[axis] = name

    def get_axis_name(self, axis):
        return self._axis_name[axis]

    def set_axis_values(self, axis, values):
        if isinstance(values, Iterable):
            self._axis_values[axis] = _obj2bytes(asarray(values))
        else:
            self._axis_values[axis] = _obj2bytes(full(self.shape[axis], values))

    def set_axis_value(self, axis, old_val, new_val):
        i = where(self._axis_values[axis] == old_val)[0][0]
        self._axis_values[axis][i] = new_val

    def get_axis_values(self, axis):
        return self._axis_values[axis]

    def item(self, *args):
        raise NotImplementedError

    def _item(self, idx, lslice_list):
        lslice = merge_lslices(lslice_list)
        idx = transform_index(idx, lslice)
        return self.item(*idx)

    def __getattr__(self, args):
        if args == self.get_axis_name(0):
            return ArrayAccessor(self, 0)
        elif args == self.get_axis_name(1):
            return ArrayAccessor(self, 1)
        raise AttributeError

    def __getitem__(self, slice_):
        if not isinstance(slice_, tuple):
            slice_ = (slice_,)

        lslice = create_lslice(slice_)
        check_lslice_boundary(lslice, self.shape)

        lslice = normalize_lslice_dim(lslice, self.ndim)
        return _create_view(self, lslice)

    @property
    def shape(self):
        raise NotImplementedError

    def _shape(self, lslice_list):
        lslice = merge_lslices(lslice_list)
        return transform_shape(self.shape, lslice)

    def _ndim(self, lslice_list):
        lslice = merge_lslices(lslice_list)
        return extract_ndim(lslice)

    @property
    def dtype(self):
        raise NotImplementedError

    def _create_array(self, lslice):
        raise NotImplementedError

    def _create_array_recurse(self, lslice_list):
        lslice = merge_lslices(lslice_list)
        lslice = normalize_lslice_dim(lslice, self.ndim)
        return self._create_array(normalize_lslice_shape(lslice, self.shape))

class ArrayNumpy(MatrixInterface):
    def __init__(self, arr):
        super(ArrayNumpy, self).__init__()
        self._arr = arr

    def item(self, *args):
        return self._arr[args]

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    def _create_array(self, lslice):
        return self._arr[lslice]

class ArrayView(ArrayViewBase):
    def __init__(self, ref, lslice):
        super(ArrayView, self).__init__()
        self._ref = ref
        self._lslice = lslice

    def set_axis_value(self, axis, old_val, new_val):
        self._ref.set_axis_value(axis, old_val, new_val)

    def get_axis_values(self, axis):
        return self._ref.get_axis_values(axis)

    @property
    def indexed_axis(self):
        return self._ref.indexed_axis

    @indexed_axis.setter
    def indexed_axis(self, axis):
        self._ref.indexed_axis = axis

    def item(self, *args):
        return self._item(args, [])

    def _item(self, idx, lslice_list):
        return self._ref._item(idx, [self._lslice]+lslice_list)

    def __getitem__(self, slice_):
        if not isinstance(slice_, tuple):
            slice_ = (slice_,)

        lslice = create_lslice(slice_)
        check_lslice_boundary(lslice, self.shape)

        lslice = normalize_lslice_dim(lslice, self.ndim)
        return _create_view(self, lslice)

    @property
    def shape(self):
        return self._shape([])

    def _shape(self, lslice_list):
        return self._ref._shape([self._lslice]+lslice_list)

    def __iter__(self):
        for i in range(self._lslice[0]):
            yield self[i].raw

    @property
    def ndim(self):
        return self._ndim([])

    def _ndim(self, lslice_list):
        return self._ref._ndim([self._lslice]+lslice_list)

    @property
    def dtype(self):
        return self._ref.dtype

    def _create_array(self, lslice):
        raise NotImplementedError

    def _create_array_recurse(self, lslice_list):
        return self._ref._create_array_recurse([self._lslice]+lslice_list)

def _create_scalar_view(ref, lslice):
    if ref.dtype is float:
        return FloatView(ref, lslice)
    if ref.dtype is bytes:
        return BytesView(ref, lslice)
    if ref.dtype is int:
        return IntView(ref, lslice)
    raise TypeError()

def _create_view(ref, lslice):
    if extract_ndim(lslice) == 0:
        return _create_scalar_view(ref, lslice)
    return ArrayView(ref, lslice)

def _obj2bytes(v):
    import numpy as np
    if isinstance(v.dtype, np.object):
        return v.astype(bytes)
    return v
