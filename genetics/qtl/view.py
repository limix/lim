from numpy import nan

from pandas import DataFrame

def qtlscan_view(qtl):
    n = len(qtl.pvalues())
    df = DataFrame({'chromid': ['unknown'] * n,
                    'position': [nan] * n,
                    'pvalue': qtl.pvalues(),
                    'effsize': qtl.candidate_effect_sizes()})
    return df
