from pandas import read_csv

from ..data import Table

def read_map(filepath):
    column_names = ['chromosome', 'snp_id', 'genetic_distance',
                    'base_pair_position']
    column_types = [bytes, bytes, float, float]

    df = read_csv(filepath, header=None, sep=r'\s+', names=column_names,
                  dtype=dict(zip(column_names, column_types)))

    table = Table(df)
    n = table.shape[0]

    cid = table['chromosome']
    sid = table['snp_id']
    table.index_values = [cid[i] + '_' + sid[i] for i in range(n)]
    table.index_name = 'marker_id'

    return table
