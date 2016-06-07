from numpy import float64, int64, bytes_

from .slice import merge_lslices
from .slice import transform_index
from .slice import transform_shape
from .slice import extract_ndim
from .slice import create_lslice
from .slice import normalize_lslice_dim
from .slice import normalize_lslice_shape
from .slice import check_lslice_boundary

class FloatView(float64):
    def __new__(cls, *_):
        return super(FloatView, cls).__new__(cls)

    def __init__(self, ref, lslice):
        self._ref = ref
        self._lslice = lslice

    def __eq__(self, rhs):
        return self.item() == rhs

    def item(self, *args):
        if len(args) == 0:
            args = (0,)
        return self._ref._item(args, [self._lslice])

    @property
    def raw(self):
        return float(self.item((0,)))

    @property
    def shape(self):
        return tuple()

    @property
    def ndim(self):
        return 0

    @property
    def dtype(self):
        return type(self.item())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.item().__str__()

class IntView(int64):
    def __new__(cls, *_):
        return super(IntView, cls).__new__(cls)

    def __init__(self, ref, lslice):
        self._ref = ref
        self._lslice = lslice

    def __eq__(self, rhs):
        return self.item() == rhs

    def item(self, *args):
        if len(args) == 0:
            args = (0,)
        return self._ref._item(args, [self._lslice])

    @property
    def raw(self):
        return int(self.item((0,)))

    @property
    def shape(self):
        return tuple()

    @property
    def ndim(self):
        return 0

    @property
    def dtype(self):
        return type(self.item())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.item().__str__()

class BytesView(bytes_):
    def __new__(cls, *_):
        return super(BytesView, cls).__new__(cls)

    def __init__(self, ref, lslice):
        super(BytesView, self).__init__()
        self._ref = ref
        self._lslice = lslice

    def __eq__(self, rhs):
        return self.item() == rhs

    def item(self, *args):
        if len(args) == 0:
            args = (0,)
        return self._ref._item(args, [self._lslice])

    @property
    def raw(self):
        return bytes(self.item((0,)))

    @property
    def shape(self):
        return tuple()

    @property
    def ndim(self):
        return 0

    @property
    def dtype(self):
        return type(self.item())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.item().__str__()

class ArrayViewBase(object):
    def __init__(self):
        pass

    def item(self, *args, **kwargs):
        raise NotImplementedError

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

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i].raw

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    def __repr__(self):
        return bytes(self._create_array_recurse([]))

    def __str__(self):
        return repr(self)

    def _create_array(self, lslice):
        raise NotImplementedError

    def _create_array_recurse(self, lslice_list):
        raise NotImplementedError

    def __array__(self):
        return self._create_array_recurse([])

class ArrayViewInterface(ArrayViewBase):
    def __init__(self):
        super(ArrayViewInterface, self).__init__()

    def item(self, *args):
        raise NotImplementedError

    def _item(self, idx, lslice_list):
        lslice = merge_lslices(lslice_list)
        idx = transform_index(idx, lslice)
        return self.item(*idx)

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

class ArrayView(ArrayViewBase):
    def __init__(self, ref, lslice):
        super(ArrayView, self).__init__()
        self._ref = ref
        self._lslice = lslice

    def item(self, *args):
        return self._item(args, [])

    def _item(self, idx, lslice_list):
        return self._ref._item(idx, [self._lslice]+lslice_list)

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
