from numpy import asarray
from numpy import loadtxt
from numpy import empty
from numpy import atleast_2d
from numpy import zeros
from numpy import concatenate
from numpy import unique
from numpy import nan
from numpy import arange

from pandas import read_csv

from .group import group
from ..data import Table
from ..data import Column
from ..data import NPyMatrix

def _read_ped_genotype(df):
    M = df.as_matrix().astype(bytes)[:,6:]
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

    df = read_csv(filepath, header=None, sep=r'\s+')

    index = arange(df.shape[0], dtype=int)

    table = Table()
    for (i, cn) in enumerate(column_names):
        c = Column(cn, index, df[i].values.astype(column_types[i]))
        table.add(c)

    fid = table['family_id']
    iid = table['individual_id']
    table.index_values = [fid[i] + '_' + iid[i] for i in index]
    table.index_name = 'sample_id'

    G = _read_ped_genotype(df)

    return (table, G)

def _read_map(filepath):
    column_names = ['chromosome', 'snp_id', 'genetic_distance',
                    'base_pair_position']
    column_types = [bytes, bytes, float, float]

    df = read_csv(filepath, header=None, sep=r'\s+')

    index = arange(df.shape[0], dtype=int)

    table = Table()
    for (i, cn) in enumerate(column_names):
        c = Column(cn, index, df[i].values.astype(column_types[i]))
        table.add(c)

    cid = table['chromosome']
    sid = table['snp_id']
    table.index_values = [cid[i] + '_' + sid[i] for i in index]
    table.index_name = 'marker_id'

    return table

def reader(basepath):
    (sample_tbl, G) = _read_ped(basepath + '.ped')
    marker_tbl = _read_map(basepath + '.map')

    NPyMatrix(G, sample_id=sample_tbl.index_values,
                 marker_id=marker_tbl.index_values)

    return (sample_tbl, marker_tbl, G)
