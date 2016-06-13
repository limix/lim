from os.path import dirname
from os.path import join
from os.path import realpath

import lim


def test_read():
    root = dirname(realpath(__file__))
    root = join(root, 'data')

    table = lim.reader.csv(join(root, '2d_array.csv'))
