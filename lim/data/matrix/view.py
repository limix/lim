from .interface import MatrixInterface

class MatrixView(MatrixInterface):
    def __init__(self, ref, index):
        super(MatrixView, self).__init__()
        self._ref = ref
        self._index = index

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
