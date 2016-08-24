# import os
#
# import h5py
#
# from numpy import array
# from numpy import asarray
# from numpy import loadtxt
# from numpy import atleast_2d
# from numpy.testing import assert_array_equal
# from numpy.testing import assert_equal
#
# import lim
#
# def _test_1d(X, R):
#     assert_array_equal(X, R)
#     assert_array_equal(X[1:], R[1:])
#     assert_array_equal(X[1:-1], R[1:-1])
#     assert_array_equal(X[-2:-1], R[-2:-1])
#     assert_equal(X.shape, R.shape)
#     assert_equal(X.ndim, R.ndim)
#     for i in range(X.shape[0]):
#         assert_equal(X[i], R[i])
#
#     assert_array_equal(asarray(X[:]), R)
#
# def _test_2d(X, R):
#     assert_array_equal(X, R)
#     assert_array_equal(X[1:,:], R[1:,:])
#     assert_array_equal(X[:,1:], R[:,1:])
#     assert_array_equal(X[1:,1:], R[1:,1:])
#     assert_array_equal(X[1:-1,:], R[1:-1,:])
#     assert_array_equal(X[1:-1,-2:-1], R[1:-1,-2:-1])
#     assert_equal(X.shape, R.shape)
#     assert_equal(X.ndim, R.ndim)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             assert_equal(X[i,j], R[i,j])
#
#     assert_equal(bytes(X), bytes(R[:]))
#
#     assert_array_equal(asarray(X[:]), R)
#
# def test_arrays():
#     root = os.path.dirname(os.path.realpath(__file__))
#     root = os.path.join(root, 'data')
#
#     fp = os.path.join(root, '1d_array_col.csv')
#     _test_2d(lim.reader.csv(fp, float),
#              atleast_2d(loadtxt(fp, delimiter=',', dtype=float)).T)
#
#     fp = os.path.join(root, '1d_array_row.csv')
#     _test_2d(lim.reader.csv(fp, float),
#              atleast_2d(loadtxt(fp, delimiter=',', dtype=float)))
#
#     fp = os.path.join(root, '1d_array_col_bytes.csv')
#     _test_1d(lim.reader.csv(fp, bytes),
#              atleast_2d(loadtxt(fp, delimiter=',', dtype=bytes)).T)
#
#     fp = os.path.join(root, '1d_array_row_bytes.csv')
#     _test_1d(lim.reader.csv(fp, bytes),
#              atleast_2d(loadtxt(fp, delimiter=',', dtype=bytes)))
#
#     fp = os.path.join(root, '2d_array.csv')
#     _test_2d(lim.reader.csv(fp, float),
#              loadtxt(fp, delimiter=',', dtype=float))
#
#     fp = os.path.join(root, '2d_array_bytes.csv')
#     _test_2d(lim.reader.csv(fp, bytes),
#              loadtxt(fp, delimiter=',', dtype=bytes))
#
#     fp = os.path.join(root, 'array.h5')
#     with h5py.File(fp, 'r') as f:
#         _test_1d(lim.reader.h5(fp, '/group/1d_array'),
#                  atleast_2d(f['/group/1d_array'][:]).T)
#
#     fp = os.path.join(root, 'array.h5')
#     with h5py.File(fp, 'r') as f:
#         _test_1d(lim.reader.h5(fp, '/group/1d_array_bytes'),
#                  atleast_2d(f['/group/1d_array_bytes']).T)
#
#     fp = os.path.join(root, 'array.h5')
#     with h5py.File(fp, 'r') as f:
#         _test_2d(lim.reader.h5(fp, '/group/2d_array'),
#                  atleast_2d(f['/group/2d_array']))
#
#     fp = os.path.join(root, 'array.h5')
#     with h5py.File(fp, 'r') as f:
#         _test_1d(lim.reader.h5(fp, '/group/2d_array_bytes'),
#                  atleast_2d(f['/group/2d_array_bytes']))
#
#     root = os.path.dirname(os.path.realpath(__file__))
#     basepath = os.path.join(root, 'data', 'plink', 'test')
#     p = lim.reader.bed(basepath)
#     R = array([[0, 3, 2, 3, 3, 3], [3, 2, 1, 3, 3, 3], [3, 1, 1, 2, 2, 0]])
#     X, individuals, markers = p[0], p[1], p[2]
#     _test_2d(X, R)
#
#     family_id = array(['1', '1', '1', '2', '2', '2'])
#     _test_1d(individuals['family_id'], family_id)
#
#     individual_id = array(['1', '2', '3', '1', '2', '3'])
#     _test_1d(individuals['individual_id'], individual_id)
#
#     paternal_id = array(['0', '0', '1', '0', '0', '1'])
#     _test_1d(individuals['paternal_id'], paternal_id)
#
#     maternal_id = array(['0', '0', '2', '0', '0', '2'])
#     _test_1d(individuals['maternal_id'], maternal_id)
#
#     sex = array(['1', '1', '1', '1', '1', '1'], bytes)
#     _test_1d(individuals['sex'], sex)
#
#     phenotype = array([-9, -9, 2, -9, 2, 2], int)
#     _test_1d(individuals['phenotype'], phenotype)
#
#     snp_ids = array(['snp1', 'snp2', 'snp3'], bytes)
#     _test_1d(markers['snp_id'], snp_ids)
#
#     genetic_dist = array([ 0.,  0.,  0.], float)
#     _test_1d(markers['genetic_dist'], genetic_dist)
#
#     chrom_ids = array(['1', '1', '1'], bytes)
#     _test_1d(markers['chrom'], chrom_ids)
#
#     bp_pos = array([1, 2, 3], int)
#     _test_1d(markers['bp_pos'], bp_pos)
