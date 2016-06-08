from numpy import asarray
from numpy import loadtxt
from numpy import empty
from numpy import zeros

from .cplink import bed_ffi
from ..data.array import ArrayViewInterface

def _read_fam(filepath):
    d = loadtxt(filepath, bytes)
    if d.shape[1] != 6:
        msg = "There are %d columns instead of six: %d." % d.shape[1]
        raise TypeError(msg)

    return dict(family_id=d[:,0], individual_id=d[:,1],
                paternal_id=d[:,2], maternal_id=d[:,3],
                sex=d[:,4], phenotype=d[:,5].astype(int))

def _read_map(filepath):
    d = loadtxt(filepath, bytes)
    chrom = d[:,0]
    snp_id = d[:,1]
    genetic_dist = asarray(d[:,2], float)
    bp_pos = asarray(d[:,3], int)
    return dict(chrom=chrom, snp_id=snp_id, genetic_dist=genetic_dist,
                bp_pos=bp_pos)


class BedPath(ArrayViewInterface):
    def __init__(self, filepath, nsnps, nindividuals):
        super(BedPath, self).__init__()
        self._filepath = filepath
        self._nsnps = nsnps
        self._nindividuals = nindividuals

    def item(self, *args):
        fp = bed_ffi.ffi.new("char[]", self._filepath)
        return bed_ffi.lib.read_item(fp, self.shape[0], self.shape[1],
                                     args[0], args[1])

    @property
    def dtype(self):
        return int

    @property
    def shape(self):
        return (self._nsnps, self._nindividuals)

    @property
    def ndim(self):
        return 2

    def _create_array(self, lslice):
        rslice, cslice = lslice[0], lslice[1]

        fp = bed_ffi.ffi.new("char[]", self._filepath)

        nrows_read = (rslice.stop - rslice.start) // rslice.step
        ncols_read = (cslice.stop - cslice.start) // cslice.step

        X = empty((nrows_read, ncols_read), dtype=int)
        pointer = bed_ffi.ffi.cast("long*", X.ctypes.data)

        bed_ffi.lib.read_slice(fp, self.shape[0], self.shape[1],
                               rslice.start, rslice.stop, rslice.step,
                               cslice.start, cslice.stop, cslice.step,
                               pointer)

        return X

def reader(basepath):
    individuals = _read_fam(basepath + '.fam')
    snps = _read_map(basepath + '.map')
    genotype = BedPath(basepath + '.bed', len(snps['snp_id']),
                       len(individuals['individual_id']))
    return (genotype, individuals, snps)
