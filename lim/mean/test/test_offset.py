import numpy as np
import numpy.testing as npt

from lim.func import check_grad

from lim.mean import OffsetMean


def test_value():
    random = np.random.RandomState(0)
    mean = OffsetMean()
    offset = random.randn()
    mean.offset = offset

    n = 10
    oarr = offset * np.ones(n)

    npt.assert_almost_equal(mean.value(n), oarr)


def test_gradient():
    random = np.random.RandomState(0)
    mean = OffsetMean()

    n = 10

    def func(x):
        mean.offset = x[0]
        return mean.value(n)

    def grad(x):
        mean.offset = x[0]
        return [mean.derivative_offset(n)]

    npt.assert_almost_equal(check_grad(func, grad, [2.0]), 0, decimal=6)
