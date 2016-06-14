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

    # d = loadtxt(filepath, bytes)
    # if d.shape[1] != 6:
    #     msg = "There are %d columns instead of six: %d." % d.shape[1]
    #     raise TypeError(msg)
    #
    # return dict(family_id=d[:,0], individual_id=d[:,1],
    #             paternal_id=d[:,2], maternal_id=d[:,3],
    #             sex=d[:,4], phenotype=d[:,5].astype(int))

# def _read_map(filepath):
#     d = loadtxt(filepath, bytes)
#     chrom = d[:,0]
#     snp_id = d[:,1]
#     genetic_dist = asarray(d[:,2], float)
#     bp_pos = asarray(d[:,3], int)
#     return dict(chrom=chrom, snp_id=snp_id, genetic_dist=genetic_dist,
#                 bp_pos=bp_pos)


class BedPath(MatrixInterface):
    def __init__(self, filepath, nsnps, nindividuals):
        super(BedPath, self).__init__()
        self._filepath = filepath
        self._nsnps = nsnps
        self._nindividuals = nindividuals

    def item(self, *args):
        fp = bed_ffi.ffi.new("char[]", self._filepath)

        index_msg_err = "Provide an integer or a pair of integers."

        if len(args) == 0:
            raise IndexError(index_msg_err)

        if len(args) == 1:
            raise NotImplementedError

        if len(args) == 2:
            return bed_ffi.lib.read_item(fp, self.shape[0], self.shape[1],
                                         args[0], args[1])

        raise IndexError(index_msg_err)

    def __getitem__(self, args):
        raise NotImplementedError

    @property
    def shape(self):
        return (self._nsnps, self._nindividuals)

    @property
    def dtype(self):
        return int

    def __repr__(self):
        return repr(self.__array__())

    def __str__(self):
        return bytes(self.__array__())

    def __array__(self, *args):
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
    #     bed_ffi.lib.read_slice(fp, self.shape[0], self.shape[1],
    #                            rslice.start, rslice.stop, rslice.step,
    #                            cslice.start, cslice.stop, cslice.step,
    #                            pointer)
    #
    #     return X

def reader(basepath):
    individuals = _read_fam(basepath + '.fam')
    # snps = read_map(basepath + '.map')
    # genotype = BedPath(basepath + '.bed', len(snps['snp_id']),
    #                    len(individuals['individual_id']))
    # return (genotype, individuals, snps)
