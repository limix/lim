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

lrt = lim.genetics.qtl.binomial_scan(nsuccesses, ntrials, X, G, progress=False)
print(lrt)
