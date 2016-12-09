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

  from lim.genetics.phenotype import NormalPhenotype
  lrt = lim.genetics.qtl.scan(NormalPhenotype(y), X, G, progress=False)
  print(lrt.pvalues())


The output should be similar to:

.. testoutput::

  [ 0.0033316   0.02921172  0.01820074  0.00701774  0.03461768  0.01167527
    0.94692018  0.81973856  0.08117947  0.86643508  0.57083216  0.69779772
    0.60996543  0.77978829  0.75514675  0.84236536  0.45432517  0.61710616
    0.48900144  0.34261678  0.35086873  0.62351578  0.71528971  0.8871672
    0.59728122  0.12720746  0.09100303  0.00853008  0.1994399   0.6647478
    0.37264595  0.72843704  0.47056861  0.99961927  0.96317648  0.93515525
    0.32407077  0.35827617  0.14577005  0.09390726  0.02432996  0.15756909
    0.70482818  0.92614646  0.59598903  0.15551889  0.80515554  0.44285868
    0.31682123  0.30309932  0.66117806  0.18398747  0.35233084  0.78545959
    0.12947138  0.76652024  0.49702157  0.71669496  0.12607443  0.23364216
    0.1708559   0.41224431  0.16092503  0.36767752  0.83013318  0.96856704
    0.17093528  0.11393999  0.27513107  0.25051797  0.67973977  0.95171302
    0.33991434  0.52360602  0.36217087  0.92046059  0.3357004   0.99753087
    0.367863    0.53227975  0.58445471  0.94780799  0.6358703   0.35140349
    0.39540056  0.68762739  0.48245462  0.16427795  0.83561238  0.91194995
    0.88152921  0.50662769  0.0928629   0.95011819  0.21613037  0.7304005
    0.61591707  0.87132833  0.16530783  0.675803  ]


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
