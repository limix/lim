===================
Lim's documentation
===================

Lim is an efficient implementation of Generalized Linear Mixed Models for
genomic analysis.

-------
Install
-------

The recommended way of installing it is via `conda`_::

  conda install -c conda-forge ncephes limix-inference
  conda install h5py pandas tabulate pytest

and then::

  pip install lim

.. _conda: http://conda.pydata.org/docs/index.html

------------
QTL Analysis
------------

We have support for Normal, Bernoulli, Binomial, and Poisson residual noises.
In other words, we can handle phenotypes that follows those aforementioned
distributions when conditioned on the latent variables.
The scan functions described bellow return an instance of
:func:`lim.genetics.qtl.LikelihoodRatioTest` from which p-values, candidate
effect sizes, log marginal likelihoods, and other statistics can be retrieved.


Continuous phenotypes
^^^^^^^^^^^^^^^^^^^^^

This example uses :func:`lim.genetics.qtl.scan` together with the
:class:`lim.genetics.phenotype.NormalPhenotype` phenotype to perform an
association scan between markers contained in `X` and the phenotype defined by
`y`, while accounting for background signal via `G`:

.. testcode::

  from lim.genetics import qtl
  from lim.genetics.phenotype import NormalPhenotype

  from numpy import random
  from numpy import zeros

  random = random.RandomState(0)
  N = 50
  P = 100

  # genetic markers
  X = random.randn(N, P)

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

  lrt = qtl.scan(NormalPhenotype(y), X, G, progress=False)
  print(lrt.pvalues())


The output should be similar to:

.. testoutput::

  [  6.87226966e-09   1.65621740e-04   1.45057979e-02   9.08631729e-03
   2.88430470e-02   1.05599085e-03   3.83901168e-01   4.75244513e-01
   2.06062424e-01   4.33645056e-01   9.99052400e-01   8.01023844e-01
   5.33902238e-01   3.03104210e-01   7.02316567e-01   5.46747256e-01

   6.07466135e-01   9.15258586e-01   3.74982775e-01   1.68710256e-01
   3.49574705e-01   7.15444096e-01   3.84825920e-01   9.12717538e-01
   6.88843100e-01   9.06313721e-01   3.54342123e-01   6.57525420e-03
   2.33376259e-01   4.53280271e-01   3.01684699e-01   9.92464728e-01
   2.40974542e-01   9.01601013e-01   4.26648639e-01   5.85426518e-01
   1.97365816e-01   9.28328495e-01   1.51583678e-01   4.24944895e-01
   1.06593762e-01   1.33648645e-01   9.40653230e-01   7.53116792e-01
   7.38882698e-01   9.89767694e-02   4.86112325e-01   9.16625928e-01
   7.28234152e-01   4.46989040e-01   9.17845023e-01   7.41522322e-01
   2.31972282e-01   6.02540180e-01   3.18359928e-01   6.98488103e-01
   7.87115649e-01   7.54821171e-01   1.96862866e-01   4.82670717e-01
   5.85543401e-01   4.01501698e-01   5.92180818e-01   6.86091889e-01
   3.37132717e-01   5.87199932e-01   1.40895638e-01   1.98146742e-02
   1.64466477e-01   3.86691215e-01   4.95486769e-01   5.34303914e-01
   1.38382652e-01   8.64088766e-01   4.17129488e-01   4.64317758e-01
   4.73011413e-01   9.31850226e-01   5.18027105e-01   2.52311113e-01
   8.72660187e-01   8.74407171e-01   5.43104679e-01   1.68190811e-01
   4.13711687e-01   6.98381079e-01   9.16977846e-01   2.27596988e-01
   7.74743294e-01   8.91415290e-01   2.28252559e-01   5.03789557e-01
   2.06060858e-01   9.64448330e-01   1.77800487e-01   5.19077492e-01
   9.40306149e-01   8.08048306e-01   8.44523318e-02   7.38634876e-01]


Count phenotypes
^^^^^^^^^^^^^^^^

This example uses :func:`lim.genetics.qtl.scan` together with the
:class:`lim.genetics.phenotype.BinomialPhenotype` phenotype to perform an
association scan between markers contained in `X` and the phenotype defined by
`y`, while accounting for background signal via `G`:

.. testcode::

  from lim.genetics import qtl
  from lim.genetics.phenotype import BinomialPhenotype

  from numpy import random
  from numpy import asarray
  from numpy import zeros
  from numpy import empty

  random = random.RandomState(0)
  N = 10
  P = 15

  # genetic markers
  X = random.randn(N, P)

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

  lrt = qtl.scan(BinomialPhenotype(nsuccesses, ntrials), X,
               G, progress=False)
  print(lrt.pvalues())

The output should be similar to:

.. testoutput::

  [ 0.94456703  0.01903082  0.02929448  0.65806501  0.62052973  0.49177804
    0.74383322  0.72037241  0.26564913  0.65786845  0.88489038  0.02357311
    0.58811967  0.19880954  0.94712376]

-----------------
Comments and bugs
-----------------

You can get the source and open issues `on Github.`_

.. _on Github.: https://github.com/limix/lim
