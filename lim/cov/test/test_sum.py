import numpy as np
import numpy.testing as npt

from ..sum import SumCov
from ..linear import LinearCov
from ...func import check_grad

def test_value():
    random = np.random.RandomState(0)
    cov_left = LinearCov()
    cov_right = LinearCov()

    X0 = random.randn(4, 20)
    X1 = random.randn(4, 15)

    cov_left.set_data(X0, X0)
    cov_right.set_data(X1, X1)

    cov = SumCov([cov_left, cov_right])
    K = cov.data('learn').value()
    npt.assert_almost_equal(K[0,0], 37.95568923)
    npt.assert_almost_equal(K[3,1], 4.53034295)

# def test_gradient():
#     random = np.random.RandomState(0)
#     cov = LinearCov()
#     cov.scale = 2.
#
#     x0 = random.randn(10)
#     x1 = random.randn(10)
#
#     def func(x):
#         cov.scale = np.exp(x[0])
#         return cov.value(x0, x1)
#
#     def grad(x):
#         cov.scale = np.exp(x[0])
#         return [cov.derivative_logscale(x0, x1)]
#
#     npt.assert_almost_equal(check_grad(func, grad, [2.0]), 0, decimal=6)
