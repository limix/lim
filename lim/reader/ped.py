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

class PED(object):
    def __init__(self, samples, markers, G):
        self._samples = samples
        self._markers = markers
        self._G = G

    def get_sample_ids(self):
        return self._samples.attr('family_id').get_axis_values(0)

    def set_sample_id(self, old_id, new_id):
        import ipdb; ipdb.set_trace()
        for k in self._samples.keys():
            self._samples.attr(k).set_axis_value(old_id, new_id)
        # return self._samples.attr('family_id').get_axis_values(0)

    def get_marker_ids(self):
        return self._samples.attr('chromosome').get_axis_values(0)

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

def reader(basepath):
    (sample_attrs, G) = _read_ped(basepath + '.ped')
    marker_attrs = _read_map(basepath + '.map')

    samples = group(sample_attrs.values(), 'attr', sample_attrs.keys())
    markers = group(marker_attrs.values(), 'attr', marker_attrs.keys())

    return PED(samples, markers, G)
