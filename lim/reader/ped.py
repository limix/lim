from numpy import unique
from numpy import concatenate
from numpy import nan
from numpy import empty

from pandas import read_csv

from .map import read_map
from ..data import Table
from ..data import NPyMatrix

def _read_ped_genotype(M):
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

    return G

def _read_ped(filepath):
    column_names = ['family_id', 'individual_id', 'paternal_id', 'maternal_id',
                    'sex', 'phenotype']
    column_types = [bytes, bytes, bytes, bytes, bytes, float]

    n = len(column_names)
    df = read_csv(filepath, header=None, sep=r'\s+', names=column_names,
                  usecols=range(n), dtype=dict(zip(column_names, column_types)))

    table = Table(df)
    n = table.shape[0]

    fid = table['family_id']
    iid = table['individual_id']
    table.index_values = [fid[i] + '_' + iid[i] for i in range(n)]
    table.index_name = 'sample_id'

    df = read_csv(filepath, header=None, sep=r'\s+')
    G = _read_ped_genotype(df.as_matrix().astype(bytes)[:,6:])

    return (table, G)

def reader(basepath):
    (sample_tbl, G) = _read_ped(basepath + '.ped')
    marker_tbl = read_map(basepath + '.map')

    NPyMatrix(G, sample_id=sample_tbl.index_values,
                 marker_id=marker_tbl.index_values)

    return (sample_tbl, marker_tbl, G)
