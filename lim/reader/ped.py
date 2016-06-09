from numpy import asarray
from numpy import loadtxt
from numpy import empty
from numpy import atleast_2d
from numpy import zeros
from numpy import concatenate
from numpy import unique
from numpy import nan

from pandas import read_csv

from .group import group

from ..data.array import ArrayNumpy

# def _read_fam(filepath):
#     d = loadtxt(filepath, bytes)
#     if d.shape[1] != 6:
#         msg = "There are %d columns instead of six: %d." % d.shape[1]
#         raise TypeError(msg)
#
#     return dict(family_id=d[:,0], individual_id=d[:,1],
#                 paternal_id=d[:,2], maternal_id=d[:,3],
#                 sex=d[:,4], phenotype=d[:,5].astype(int))
#
# def _read_map(filepath):
#     d = loadtxt(filepath, bytes)
#     chrom = d[:,0]
#     snp_id = d[:,1]
#     genetic_dist = asarray(d[:,2], float)
#     bp_pos = asarray(d[:,3], int)
#     return dict(chrom=chrom, snp_id=snp_id, genetic_dist=genetic_dist,
#                 bp_pos=bp_pos)
#
#
# class Ped(ArrayViewInterface):
#     def __init__(self, filepath, nsnps, nindividuals):
#         super(BedPath, self).__init__()
#         self._filepath = filepath
#         self._nsnps = nsnps
#         self._nindividuals = nindividuals
#
#     def item(self, *args):
#         fp = bed_ffi.ffi.new("char[]", self._filepath)
#         return bed_ffi.lib.read_item(fp, self.shape[0], self.shape[1],
#                                      args[0], args[1])
#
#     @property
#     def dtype(self):
#         return int
#
#     @property
#     def shape(self):
#         return (self._nsnps, self._nindividuals)
#
#     @property
#     def ndim(self):
#         return 2
#
#     def _create_array(self, lslice):
#         rslice, cslice = lslice[0], lslice[1]
#
#         fp = bed_ffi.ffi.new("char[]", self._filepath)
#
#         nrows_read = (rslice.stop - rslice.start) // rslice.step
#         ncols_read = (cslice.stop - cslice.start) // cslice.step
#
#         X = empty((nrows_read, ncols_read), dtype=int)
#         pointer = bed_ffi.ffi.cast("long*", X.ctypes.data)
#
#         bed_ffi.lib.read_slice(fp, self.shape[0], self.shape[1],
#                                rslice.start, rslice.stop, rslice.step,
#                                cslice.start, cslice.stop, cslice.step,
#                                pointer)
#
#         return X

# Family ID
# Individual ID
# Paternal ID
# Maternal ID
# Sex (1=male; 2=female; other=unknown)
# Phenotype

def _read_ped(filepath):
    df = read_csv(filepath, header=None, sep=r'\s+')
    M = df.as_matrix().astype(bytes)
    sample_attrs = dict(family_id=M[:,0], individual_id=M[:,1],
                        paternal_id=M[:,2], maternal_id=M[:,3], sex=M[:,4],
                        phenotype=M[:,5])

    M = M[:,6:]
    nsnps = M.shape[1] // 2
    nsamples = M.shape[0]

    G = empty((nsamples, nsnps))
    for i in range(nsnps):
        left  = M[:,i*2 + 0]
        right = M[:,i*2 + 1]

        v = concatenate([left, right])
        u = unique(v)
        a = list(set(u).difference('0'))
        if len(a) == 0 or len(a) > 2:
            raise ValueError

        if len(a) == 1:
            minor_allele = a[0]
        else:
            a0 = sum(v == a[0])
            a1 = sum(v == a[1])
            minor_allele = a[0] if a0 <= a1 else a[1]

        G[:,i]  = left == minor_allele
        G[:,i] += right == minor_allele
        G[left == '0',i] = nan

    samples = dict()
    for (k, v) in iter(sample_attrs.items()):
        arr = ArrayNumpy(atleast_2d(v).T)

        arr.set_axis_name(0, 'sample_id')
        fa = sample_attrs['family_id']
        ia = sample_attrs['individual_id']
        values = asarray([fa[i] + '_' + ia[i] for i in range(len(fa))])
        arr.set_axis_values(0, values)

        samples[k] = arr
    return (samples, G)

def _read_map(filepath):
    df = read_csv(filepath, header=None, sep=r'\s+')
    M = df.as_matrix()
    chrom = M[:,0].astype(bytes)
    snp_id = M[:,1].astype(bytes)
    genetic_dist = M[:,2].astype(float)
    bp_pos = M[:,3].astype(int)
    return dict(chromosome=chrom, snp_id=snp_id,
                genetic_distance=genetic_dist,
                base_pair_position=bp_pos)

# def _read_map(filepath):
#     d = loadtxt(filepath, bytes)
#     chrom = d[:,0]
#     snp_id = d[:,1]
#     genetic_dist = asarray(d[:,2], float)
#     bp_pos = asarray(d[:,3], int)
#     return dict(chrom=chrom, snp_id=snp_id, genetic_dist=genetic_dist,
#                 bp_pos=bp_pos)

def reader(basepath):
    (sample_attrs, G) = _read_ped(basepath + '.ped')
    marker_attrs = _read_map(basepath + '.map')

    samples = group(sample_attrs.values(), 'attr', sample_attrs.keys())
    markers = group(marker_attrs.values(), 'attr', marker_attrs.keys())

    return (samples, markers, G)
