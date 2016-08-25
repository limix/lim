from numpy import asarray
from numpy import errstate
from numpy import isnan

def stdnorm(X):
    X = asarray(X, float)
    m = X.mean(0)
    s = X.std(0)
    with errstate(invalid='ignore'):
        r = (X - m) / s
    r[isnan(r)] = 0
    return r
