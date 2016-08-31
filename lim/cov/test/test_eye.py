import numpy.testing as npt
from numpy.random import RandomState
from numpy import exp

from lim.cov import EyeCov
from lim.util.fruits import Oranges
from lim.util.fruits import Apples
from lim.func import check_grad


def test_eye_value():
    cov = EyeCov()
    cov.scale = 2.1
    o = Oranges(None)
    npt.assert_almost_equal(2.1, cov.value(o, o))


def test_eye_gradient_1():
    cov = EyeCov()
    cov.scale = 2.1
    a = Apples(None)
    cov.set_data((a, a))

    def func(x):
        cov.scale = exp(x[0])
        return cov.data().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.data().gradient()

    npt.assert_almost_equal(check_grad(func, grad, [0.1]), 0)


def test_eye_gradient_2():
    cov = EyeCov()
    cov.scale = 2.1
    a = Apples(5)
    cov.set_data((a, a))

    def func(x):
        cov.scale = exp(x[0])
        return cov.data().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.data().gradient()

    npt.assert_almost_equal(check_grad(func, grad, [0.1]), 0)


def test_eye_gradient_3():
    cov = EyeCov()
    cov.scale = 2.1
    random = RandomState()
    a = Apples(5)
    o = Oranges(4)
    cov.set_data((a, o))

    def func(x):
        cov.scale = exp(x[0])
        return cov.data().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.data().gradient()

    npt.assert_almost_equal(check_grad(func, grad, [0.1]), 0)
