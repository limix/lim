import lim

from numpy.random import RandomState
from numpy import sqrt

random = RandomState(0)
N = 50

G0 = random.randn(N, 100)
G0 = lim.tool.normalize.stdnorm(G0)
G0 /= sqrt(G0.shape[1])

G1 = random.randn(N, 150)
G1 = lim.tool.normalize.stdnorm(G1)
G1 /= sqrt(G1.shape[1])

G2 = random.randn(N, 200)
G2 = lim.tool.normalize.stdnorm(G2)
G2 /= sqrt(G2.shape[1])


u = 0.8 * random.randn(150)
y = G1.dot(u) + 0.2 * random.randn(N)

var = lim.genetics.variance.normal_decomposition(y, [G0, G1, G2])
print(var)
# var = lim.genetics.variance.normal_decomposition(y, dict(Genetic=G0,
#                                                             Cage=G1,
#                                                             Weather=G2))
# var = lim.genetics.variance.normal_decomposition(y, dict(Genetic=(G0, False),
#                                                             Cage=(K1, True),
# Weather=(K2, True)))
