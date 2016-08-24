from .interface import MatrixInterface

class MatrixView(MatrixInterface):
    def __init__(self, ref, index):
        super(MatrixView, self).__init__()
        self._ref = ref
        self._index = index

    def item(self, *args):
        return self._self(args, [])

    def _item(self, idx, index_list):
        return self._ref._item(idx, [self._index]+index_list)

    def __getitem__(self, args):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def _shape(self, index_list):
        return self._shape([self._index]+index_list)

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return self._ref.dtype

    def __repr__(self):
        return repr(self.__array__())

    def __str__(self):
        return bytes(self.__array__())

    def __array__(self, *args, **kwargs):
        kwargs = dict(kwargs)

        if 'index_list' not in kwargs:
            kwargs['index_list'] = []

        kwargs['index_list'] = [self._index] + kwargs['index_list']
        return self._ref.__array__(*args, **kwargs)

    @property
    def sample_ids(self):
        return self._sample_ids([])

    def _sample_ids(self, index_list):
        return self._sample_ids([self._index]+index_list)

    @property
    def marker_ids(self):
        return self._marker_ids([])

    def _marker_ids(self, index_list):
        return self._marker_ids([self._index]+index_list)


    # def item(self, *args):
    #     return self._item(args, [])
    #
    # def _item(self, idx, lslice_list):
    #     return self._ref._item(idx, [self._lslice]+lslice_list)
    #
    # def __getitem__(self, slice_):
    #     if not isinstance(slice_, tuple):
    #         slice_ = (slice_,)
    #
    #     lslice = create_lslice(slice_)
    #     check_lslice_boundary(lslice, self.shape)
    #
    #     lslice = normalize_lslice_dim(lslice, self.ndim)
    #     return _create_view(self, lslice)
    #
    # @property
    # def shape(self):
    #     return self._shape([])
    #
    # def _shape(self, lslice_list):
    #     return self._ref._shape([self._lslice]+lslice_list)
    #
    # def __iter__(self):
    #     for i in range(self._lslice[0]):
    #         yield self[i].raw
    #
    # @property
    # def ndim(self):
    #     return self._ndim([])
    #
    # def _ndim(self, lslice_list):
    #     return self._ref._ndim([self._lslice]+lslice_list)
    #
    # @property
    # def dtype(self):
    #     return self._ref.dtype
    #
    # def _create_array(self, lslice):
    #     raise NotImplementedError
    #
    # def _create_array_recurse(self, lslice_list):
    #     return self._ref._create_array_recurse([self._lslice]+lslice_list)
