from numpy import ones
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from lim.func import check_grad
from lim.func.check_grad import approx_fprime

from lim.mean import LinearMean


def test_value():
    random = RandomState(0)
    mean = LinearMean(5)
    effsizes = random.randn(5)
    mean.effsizes = effsizes

    x = random.randn(5)
    assert_almost_equal(mean.value(x), 2.05989432128)


def test_gradient():
    random = RandomState(1)
    mean = LinearMean(5)
    effsizes = random.randn(5)
    mean.effsizes = effsizes

    x = random.randn(5)

    def func(x0):
        mean.effsizes[0] = x0[0]
        return mean.value(x)

    def grad(x0):
        mean.effsizes[0] = x0[0]
        return [mean.derivative_offset(x)]

    # ndarray_listener
    # print(func([1.2]))
    # print(grad([1.2]))
    # assert_almost_equal(check_grad(func, grad, [1.2]), 0, decimal=6)
    # from numpy import array
    # # print(approx_fprime([1.2], func, 1e-4))
    # import pytest
    # pytest.set_trace()
    mean.effsizes[0] = 1.0
    # print(mean.effsizes)
    # print(mean.value(array([1.5, 0, 0, 0, 0])))
    # mean.effsizes[0] = 2.0
    # print(mean.effsizes)
    # print(mean.value(array([1.5, 0, 0, 0, 0])))
