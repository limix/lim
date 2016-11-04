from numpy import copyto

def gower_normalization(K, out=None):
    """Perform Gower normalizion on covariance matrix K.

    The rescaled covariance matrix has sample variance of 1.
    """
    c = (K.shape[0] - 1) / (K.trace() - K.mean(0).sum())
    if out is None:
        return c * K

    copyto(out, K)
    out *= c
