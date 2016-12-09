QTL Analysis
------------

We have support for Normal, Bernoulli, Binomial, and Poisson phenotypes
(i.e., residual noise).
The scan functions described bellow return an instance of
:func:`lim.genetics.qtl.LikelihoodRatioTest` from which p-values, candidate
effect sizes, log marginal likelihoods, and other statistics can be retrieved.


Continuous phenotypes
^^^^^^^^^^^^^^^^^^^^^

This example uses :func:`lim.genetics.qtl.normal_scan` to perform an
association scan between markers contained in `X` and the phenotype defined by
`y`, while accounting for background signal via `G`:

.. testcode::

  import lim

  from numpy import random
  from numpy import ones
  from numpy import zeros
  from numpy import sqrt
  from numpy import ones

  random = random.RandomState(0)
  N = 50
  P = 100

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

  offset = 0.4

  # phenotype definition
  y = offset + X.dot(u) + 0.2 * random.randn(N)

  G = X[:, 2:].copy()

  lrt = lim.genetics.qtl.normal_scan(y, X, G, progress=False)
  print(lrt)


The output should be similar to:

#.. program-output:: python ../examples/normal_qtl_analysis.py


Count phenotypes
^^^^^^^^^^^^^^^^

This example uses :func:`lim.genetics.qtl.binomial_scan` to perform an
association scan between markers contained in `X` and the phenotype defined by
`y`, while accounting for background signal via `G`:

.. literalinclude:: /../examples/binomial_qtl_analysis.py

The output should be similar to::

    Null model:

        Phenotype:
            y_i = ∑_{j=1}^{n_i} 𝟏(f_i + ε_{i,j} > 0)

        Latent phenotype:
            f_i = o_i + u_i + e_i

        Definitions:
            M      : covariates
            o      : fixed-effects signal = M [ 2.3302].T
            u      : background signal    ~ Normal(0,   4.9600 * Kinship)
            e      : environmental signal ~ Normal(0,   1.2388 * I)
            ε_{i,j}: instrumental signal  ~ Normal(0,   1)
            n_i    : number of draws of the i-th sample

        Model statistics:
            Covariate effect sizes: [ 2.3302]
            Fixed-effect variances:  0.0000
            Genetic variance:        4.9600
            Environmental variance:  1.2388
            Instrumental variance:   1.0000
            Heritability:            0.8002
            Genetic ratio:           0.6890
            Noise ratio:             0.5533

    Alternative model:

        Latent phenotype:
            f_i = o_i + b_j x_{i,j} + u_i + e_i

        Definitions:
            b_j    : effect-size of the j-th candidate marker
            x_{i,j}: j-th candidate marker of the i-th sample


        Candidate effect sizes:
                 Min         1Q      Median        3Q       Max
            -1.19916  -0.314468  -0.0183151  0.294666  0.964101

        Candidate log marginal likelihoods:
                Min        1Q    Median        3Q       Max
            -77.904  -77.8699  -77.6554  -77.2389  -73.7476

        Candidate p-values:
                     Min            1Q        Median            3Q           Max
            3.936413e-03  2.487912e-01  4.807105e-01  7.939106e-01  9.975996e-01
