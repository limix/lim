class MatrixInterface(object):
    def __init__(self):
        pass

    def item(self, *args):
        raise NotImplementedError

    def __getitem__(self, args):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        raise NotImplementedError

    def __repr__(self):
        return repr(self.__array__())

    def __str__(self):
        return bytes(self.__array__())

    def __array__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_ids(self):
        raise NotImplementedError

    @property
    def marker_ids(self):
        raise NotImplementedError
