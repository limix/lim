from os.path import dirname
from os.path import join
from os.path import realpath

from numpy import array
from numpy.testing import assert_array_equal

import lim


def test_read():
    root = dirname(realpath(__file__))
    root = join(root, 'data')

    table = lim.reader.h5(join(root, 'array.h5'), '/group/2d_array')

    R = [[ 0.,  1.,  2.,  1.,  0.],
         [ 0.,  1.,  2.,  1.,  0.],
         [ 1.,  0.,  1.,  1.,  1.],
         [ 2.,  2.,  0.,  1.,  0.]]

    assert_array_equal(table.as_matrix(), array(R))
