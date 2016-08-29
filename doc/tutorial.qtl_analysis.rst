QTL Analysis
============

The first example uses :func:`lim.genetics.qtl.normal_scan` to perform an
association scan between markers contained in `X` and the phenotype defined by
`y`, while account for background signal via `G`::

    import lim

    from numpy import random
    from numpy import zeros
    from numpy import sqrt
    from numpy import ones

    random = random.RandomState(0)
    N = 500
    P = 900

    # genetic markers
    X = random.randn(N, P)
    X = lim.tool.normalize.stdnorm(X)
    X /= sqrt(X.shape[1])

    # effect sizes
    u = zeros(P)
    u[0] = +1
    u[1] = +1
    u[2] = -1
    u[3] = -1
    u[4] = -1
    u[5] = +1

    offset = 0.8

    # phenotype definition
    y = offset + X.dot(u) + 0.9 * random.randn(N)

    G = X[:,3:].copy()

    info = lim.genetics.qtl.normal_scan(y, X, G, verbose=False)

The output should be similar to::

    Null model learning...............................................
    Alternative model learning 100%
    ┌-------------------------------- INPUT INFO ---------------------------------┐
    |Likelihood                  Normal                                           |
    |Phenotype mean              0.808466746969                                   |
    |Phenotype std               0.834680611419                                   |
    |Phenotype (min, max)        (-1.445940606416142, 3.1803927883977563)         |
    |Background data             provided via markers                             |
    |# background markers        897                                              |
    |# const background markers  0                                                |
    |Kinship diagonal mean       1.0                                              |
    |Covariates                  a single column of ones                          |
    |Kinship rank                499                                              |
    |# candidate markers         900                                              |
    └-----------------------------------------------------------------------------┘
    ┌-------------------------------- NULL MODEL ---------------------------------┐
    |Phenotype:                                                                   |
    |  y_i = o_i + u_i + e_i                                                      |
    |                                                                             |
    |Definitions:                                                                 |
    |  M: covariates                                                              |
    |  o: fixed-effects signal = M [ 0.808].T                                     |
    |  u: background signal    ~ Normal(0,  0.0000 * Kinship)                     |
    |  e: environmental signal ~ Normal(0,  0.6967 * I)                           |
    |                                                                             |
    |Log marginal likelihood: -619.116448                                         |
    |                                                                             |
    |Statistics (latent space):                                                   |
    |  Total variance:          0.6967     ν_o + ν_u + ν_e                        |
    |  Fixed-effect variances:  0.0000     ν_o                                    |
    |  Heritability:            0.0000     ν_u / (ν_o + ν_u + ν_e)                |
    |where ν_x is the variance of signal x                                        |
    └-----------------------------------------------------------------------------┘
    ┌----------------------------- ALTERNATIVE MODEL -----------------------------┐
    |Phenotype:                                                                   |
    |  y_i = o_i + x_j β_j + u_i + e_i                                            |
    |                                                                             |
    |Definitions:                                                                 |
    |  β_j: fixed-effect sizes of the j-th marker candidate                       |
    └-----------------------------------------------------------------------------┘
