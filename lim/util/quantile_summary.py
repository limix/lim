from numpy import asarray
from numpy import median
from numpy import percentile
from tabulate import tabulate


def quantile_summary(v, floatfmt="g"):
    v = asarray(v)

    vmin = v.min()
    v1q = percentile(v, 25)
    medi = median(v)
    v3q = percentile(v, 75)
    vmax = v.max()

    headers = ('Min', '1Q', 'Median', '3Q', 'Max')
    return tabulate([[vmin, v1q, medi, v3q, vmax]], headers=headers,
                    tablefmt="plain", floatfmt=floatfmt)
