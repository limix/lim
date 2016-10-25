from numpy import ones

def offset_covariate(covariates, n):
    return ones((n, 1)) if covariates is None else covariates
