import numpy.testing as npt
from numpy.random import RandomState
from numpy import exp

from ..eye import EyeCov
from ...func import check_grad

def test_eye_value():
    cov = EyeCov()
    cov.scale = 2.1
    random = RandomState()
    v = random.randn(10)
    npt.assert_almost_equal(2.1, cov.value(v, v))

def test_eye_gradient_1():
    cov = EyeCov()
    cov.scale = 2.1
    random = RandomState()
    v = random.randn(10)
    cov.set_data((v, v))

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
    random = RandomState()
    v = random.randn(10, 5)
    cov.set_data((v, v))

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
    v0 = random.randn(10, 5)
    v1 = random.randn(10, 5)
    cov.set_data((v0, v1))

    def func(x):
        cov.scale = exp(x[0])
        return cov.data().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.data().gradient()

    npt.assert_almost_equal(check_grad(func, grad, [0.1]), 0)
