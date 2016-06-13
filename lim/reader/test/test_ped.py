from os.path import dirname
from os.path import join
from os.path import realpath

from numpy import array
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_string_equal

import lim


def test_read():
    root = dirname(realpath(__file__))
    root = join(root, 'data')

    (stable, mtable, G) = lim.reader.ped(join(root, 'plink', 'test'))
    print(stable)
    print(mtable)

    assert_string_equal(stable['family_id']['1_1'], '1')
    assert_string_equal(stable['family_id']['2_1'], '2')
    assert_string_equal(stable['family_id']['1_2'], '1')

    assert_string_equal(stable['individual_id']['1_2'], '2')
    assert_string_equal(stable['individual_id']['2_1'], '1')

    assert_equal(mtable['base_pair_position']['1_snp1'], 1.0)
    assert_equal(mtable['base_pair_position']['1_snp2'], 2.0)
    assert_equal(mtable['base_pair_position']['1_snp3'], 3.0)

    # R = [[ 0.,  1.,  2.,  1.,  0.],
    #      [ 0.,  1.,  2.,  1.,  0.],
    #      [ 1.,  0.,  1.,  1.,  1.],
    #      [ 2.,  2.,  0.,  1.,  0.]]
    #
    # assert_array_equal(table.as_matrix(), array(R))
