import lim

from numpy import random
from numpy import ones
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

G = X[:, 3:].copy()

lrt = lim.genetics.qtl.normal_scan(y, X, G, progress=False)
print(lrt)
