from os.path import dirname
from os.path import join
from os.path import realpath

from numpy import array
from numpy import asarray
from numpy import nan
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_string_equal

import lim


def test_read():
    root = dirname(realpath(__file__))
    root = join(root, 'data')

    (stable, mtable, G) = lim.reader.bed(join(root, 'plink', 'test'))

    assert_array_equal(G.shape, (6, 3))
    # g = G[0,:]
    # print(g)
    # assert_array_equal(G[0,:], array([ 0.,  2.,  2.]))
    # assert_array_equal(G[1,:], array([  2.,  nan,   1.]))
    # assert_array_equal(G[2,:], array([ nan,   1.,   1.]))
    # assert_array_equal(G[3,:], array([  2.,   2.,  nan]))
    # assert_array_equal(G[4,:], array([  2.,   2.,  nan]))
    # assert_array_equal(G[5,:], array([ 2.,  2.,  0.]))
    #
    # assert_array_equal(G[:,0], array([  0.,   2.,  nan,   2.,   2.,   2.]))
    # assert_array_equal(G[:,1], array([  2.,  nan,   1.,   2.,   2.,   2.]))
    # assert_array_equal(G[:,2], array([  2.,   1.,   1.,  nan,  nan,   0.]))
    #
    # assert_string_equal(stable['family_id']['1_1'], '1')
    # assert_string_equal(stable['family_id']['2_1'], '2')
    # assert_string_equal(stable['family_id']['1_2'], '1')
    #
    # assert_string_equal(stable['individual_id']['1_2'], '2')
    # assert_string_equal(stable['individual_id']['2_1'], '1')
    #
    # assert_equal(mtable['base_pair_position']['1_snp1'], 1.0)
    # assert_equal(mtable['base_pair_position']['1_snp2'], 2.0)
    # assert_equal(mtable['base_pair_position']['1_snp3'], 3.0)
    #
    # R = array([[ 0.,  2.,  2.],
    #            [  2.,  nan,   1.],
    #            [ nan,   1.,   1.],
    #            [  2.,   2.,  nan],
    #            [  2.,   2.,  nan],
    #            [ 2.,  2.,  0.]])
    # assert_array_equal(asarray(G), R)
