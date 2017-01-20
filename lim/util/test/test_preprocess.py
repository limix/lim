from numpy import array, nan
from numpy.testing import assert_allclose

from lim.util.preprocess import quantile_gaussianize


def test_quantile_gaussianize():
    x = array([nan, 2.1, 2.4, -100.2, 2.2, 1.0, 0.0, 2.1])
    y = quantile_gaussianize(x)
    assert_allclose(y, [
        nan, 0.15731068461, 1.150349380376, -1.150349380376, 0.674489750196,
        -0.318639363964, -0.674489750196, 0.15731068461
    ])
