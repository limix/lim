def logbinom(k, n):
    from scipy.special import gammaln
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

if __name__ == '__main__':
    import numpy as np
    print(logbinom(4, 10))
    print(np.exp(logbinom(4, 10)))
    import scipy as sp
    print(sp.special.binom(10, 4))
