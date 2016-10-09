from numpy import ones
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from lim.func import check_grad

from lim.mean import LinearMean


def test_value():
    random = RandomState(0)
    mean = LinearMean(5)
    effsizes = random.randn(5)
    mean.effsizes = effsizes

    x = random.randn(5)
    assert_almost_equal(mean.value(x), 2.05989432128)

#     n = 10
#     oarr = effsizes * ones(n)
#
#     assert_almost_equal(mean.value(n), oarr)
#
#
# def test_gradient():
#     mean = LinearMean()
#
#     n = 10
#
#     def func(x):
#         mean.effsizes = x[0]
#         return mean.value(n)
#
#     def grad(x):
#         mean.effsizes = x[0]
#         return [mean.derivative_offset(n)]
#
#     assert_almost_equal(check_grad(func, grad, [2.0]), 0, decimal=6)
