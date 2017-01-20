from __future__ import division
from __future__ import absolute_import

from numpy import empty_like
from numpy import isfinite

from scipy.stats import rankdata
from scipy.stats import norm

def quantile_gaussianize(x):
    ok = isfinite(x)
    x[ok] *= -1
    y = empty_like(x)
    y[ok] = rankdata(x[ok])
    y[ok] = norm.isf(y[ok] / (sum(ok) + 1))
    y[~ok] = x[~ok]
    return y

if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(precision=12)
    x = np.array([np.nan, 2.1, 2.4, -100.2, 2.2, 1.0, 0.0, 2.1])
    print(x)
    y = quantile_gaussianize(x)
    print(y)
