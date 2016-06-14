from numpy import asarray
from numpy import loadtxt
from numpy import empty
from numpy import zeros

from pandas import read_csv

from .cplink import bed_ffi
from .map import read_map
from ..data import MatrixInterface
from ..data import Table

def _read_fam(filepath):
    column_names = ['family_id', 'individual_id', 'paternal_id', 'maternal_id',
                    'sex', 'phenotype']
    column_types = [bytes, bytes, bytes, bytes, bytes, float]


    df = read_csv(filepath, header=None, sep=r'\s+', names=column_names,
                  dtype=dict(zip(column_names, column_types)))
    table = Table(df)

    fid = table['family_id']
    iid = table['individual_id']
    n = table.shape[0]
    table.index_values = [fid[i] + '_' + iid[i] for i in range(n)]
    table.index_name = 'sample_id'

    return table

class BedPath(MatrixInterface):
    def __init__(self, filepath, nsamples, nmarkers):
        super(BedPath, self).__init__()
        self._filepath = filepath
        self._nmakers = nmarkers
        self._nsamples = nsamples

    def item(self, *args):
        fp = bed_ffi.ffi.new("char[]", self._filepath)

        index_msg_err = "Provide an integer or a pair of integers."

        if len(args) == 0:
            raise IndexError(index_msg_err)

        if len(args) == 1:
            raise NotImplementedError

        if len(args) == 2:
            return bed_ffi.lib.bed_read_item(fp, self.shape[0], self.shape[1],
                                             args[0], args[1])

        raise IndexError(index_msg_err)

    def __getitem__(self, args):
        raise NotImplementedError

    @property
    def shape(self):
        return (self._nmakers, self._nsamples)

    @property
    def dtype(self):
        return int

    def __repr__(self):
        return repr(self.__array__())

    def __str__(self):
        return bytes(self.__array__())

    def __array__(self, *args):
        if len(args) == 0:
            return
        import ipdb; ipdb.set_trace()
        raise NotImplementedError

    @property
    def sample_ids(self):
        raise NotImplementedError

    @property
    def marker_ids(self):
        raise NotImplementedError

    # def _create_array(self, lslice):
    #     rslice, cslice = lslice[0], lslice[1]
    #
    #     fp = bed_ffi.ffi.new("char[]", self._filepath)
    #
    #     nrows_read = (rslice.stop - rslice.start) // rslice.step
    #     ncols_read = (cslice.stop - cslice.start) // cslice.step
    #
    #     X = empty((nrows_read, ncols_read), dtype=int)
    #     pointer = bed_ffi.ffi.cast("long*", X.ctypes.data)
    #
    #     bed_ffi.lib.bed_read_slice(fp, self.shape[0], self.shape[1],
    #                            rslice.start, rslice.stop, rslice.step,
    #                            cslice.start, cslice.stop, cslice.step,
    #                            pointer)
    #
    #     return X

def reader(basepath):
    sample_tbl = _read_fam(basepath + '.fam')
    marker_tbl = read_map(basepath + '.map')
    G = BedPath(basepath + '.bed', sample_tbl.shape[0], marker_tbl.shape[0])
    return (sample_tbl, marker_tbl, G)
