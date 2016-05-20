from __future__ import division

import logging
from math import sqrt

from limix_math.linalg import qs_decomposition

def eigen_design_matrix(G):
    logger = logging.getLogger(__name__)

    logger.info('Genetic markers normalization.')
    G = G - G.mean(0)
    s = G.std(0)
    ok = s > 0.
    G[:,ok] /= s[ok]
    G /= sqrt(G.shape[1])

    logger.info('Computing the economic eigen decomposition.')

    return qs_decomposition(G)
