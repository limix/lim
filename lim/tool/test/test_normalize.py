from numpy import ones
from numpy.random import RandomState
from numpy.testing import assert_allclose

from lim.tool.normalize import stdnorm


def test_stdnorm():
    random = RandomState(38943)
    x = random.randn(10)
    X = random.randn(10, 5)
    x = stdnorm(x)
    X = stdnorm(X)

    assert_allclose(x.mean(0), [0], atol=1e-7)
    assert_allclose(x.std(0), 1, atol=1e-7)

    assert_allclose(X.mean(0), [0]*5, atol=1e-7)
    assert_allclose(X.std(0), [1]*5, atol=1e-7)

    x = ones(10)
    X = random.randn(10, 5)
    X[:,0] = 1

    assert_allclose(stdnorm(x).mean(0), [0])
    assert_allclose(stdnorm(x).std(0), [0])


if __name__ == '__main__':
    import os
    folder = os.path.dirname(os.path.realpath(__file__))
    import pytest
    pytest.main(['-x', folder, '-s'])
