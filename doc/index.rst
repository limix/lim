===================
Lim's documentation
===================

Lim is an efficient implementation of Generalized Linear Mixed Models for
genomic analysis.

*******
Install
*******

The recommended way of installing it is via `conda`_::

  conda install -c conda-forge lim

An alternative way would be via pip::

  pip install lim

.. _conda: http://conda.pydata.org/docs/index.html

************
QTL Analysis
************

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
  X = lim.tool.normalize.stdnorm(X, axis=0)
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

.. testcode::

  import lim

  from numpy import random
  from numpy import asarray
  from numpy import zeros
  from numpy import empty
  from numpy import ones
  from numpy import sqrt
  from numpy import ones

  random = random.RandomState(0)
  N = 50
  P = 100

  # genetic markers
  X = random.randn(N, P)
  X = lim.tool.normalize.stdnorm(X, axis=0)
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

  # latent phenotype definition
  f = offset + X.dot(u) + 0.2 * random.randn(N)

  # phenotype definition
  nsuccesses = empty(N)
  ntrials = random.randint(1, 30, N)
  for i in range(N):
      nsuccesses[i] = sum(f[i] > 0.2 * random.randn(ntrials[i]))
  ntrials = asarray(ntrials, float)

  G = X[:, 2:].copy()

  from lim.genetics.phenotype import BinomialPhenotype
  lrt = lim.genetics.qtl.scan(BinomialPhenotype(nsuccesses, ntrials), X,
                              G, progress=False)
  print(lrt.pvalues())

The output should be similar to:

.. testoutput::

  [ 0.01941533  0.05974973  0.22287607  0.12196036  0.00390464  0.05484215
    0.73410739  0.77561839  0.02139017  0.37770498  0.38665833  0.42453626
    0.54323949  0.93475895  0.60918312  0.89924375  0.88113106  0.49228679
    0.68271584  0.374527    0.94550831  0.72927318  0.85459755  0.91193689
    0.75023152  0.17971294  0.01314011  0.01941229  0.31704706  0.86447582
    0.61602016  0.51567901  0.13453806  0.81132991  0.87330082  0.6095185
    0.67192862  0.23207296  0.39602648  0.06313886  0.06008298  0.58746426
    0.82310481  0.26534184  0.45359096  0.36038528  0.56077226  0.2152736
    0.2502973   0.25361016  0.3827223   0.36221456  0.30415115  0.40922751
    0.38122384  0.70966208  0.12365265  0.86024364  0.22792395  0.41876851
    0.14306838  0.91980698  0.32779147  0.45793564  0.79928185  0.43292091
    0.10158896  0.63442848  0.20173139  0.19715465  0.62092913  0.90962452
    0.35988164  0.2692583   0.65899755  0.99096715  0.83528285  0.96926421
    0.7062866   0.15391244  0.93020241  0.59675382  0.59728103  0.1798022
    0.76862858  0.9121716   0.47676206  0.91313978  0.9609639   0.48296364
    0.65658776  0.88089504  0.01616766  0.67807704  0.11466733  0.71584291
    0.96650256  0.98655773  0.45722517  0.98681809]

*****************
Comments and bugs
*****************

You can get the source and open issues `on Github.`_

.. _on Github.: https://github.com/glimix/lim
