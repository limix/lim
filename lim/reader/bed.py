from pandas import read_csv

from .cplink import read_item
from .cplink import read_row_slice
from .cplink import read_col_slice
from .cplink import read

from .map import read_map

from ..data import Slice
from ..data import MatrixInterface
from ..data import normalize_access
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
        self._nsamples = nsamples
        self._nmakers = nmarkers

    def item(self, *args):
        index_msg_err = "Provide an integer or a pair of integers."

        if len(args) == 0:
            raise IndexError(index_msg_err)

        if len(args) == 1:
            raise NotImplementedError

        if len(args) == 2:
            return read_item(self._filepath, args[0], args[1], self.shape)

        raise IndexError(index_msg_err)

    def __getitem__(self, args):
        mslice = normalize_access(args, self.shape)
        if isinstance(mslice[0], int) and isinstance(mslice[1], Slice):
            return read_row_slice(self._filepath, mslice[0], mslice[1].start,
                                  mslice[1].stop, mslice[1].step, self.shape)
        elif isinstance(mslice[1], int) and isinstance(mslice[0], Slice):
            return read_col_slice(self._filepath, mslice[1], mslice[0].start,
                                  mslice[0].stop, mslice[0].step, self.shape)
        # return read_mslice(mslice)
        raise NotImplementedError

    @property
    def shape(self):
        return (self._nsamples, self._nmakers)

    @property
    def dtype(self):
        return int

    def __repr__(self):
        return repr(self.__array__())

    def __str__(self):
        return bytes(self.__array__())

    def __array__(self, *args):
        if len(args) == 0:
            return read(self._filepath, self.shape)
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
