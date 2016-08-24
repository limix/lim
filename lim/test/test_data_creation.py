# from os.path import join
# from os.path import dirname
# from os.path import realpath
#
# from numpy import array
# from numpy.testing import assert_array_equal
# from numpy.testing import assert_array_almost_equal
#
# import lim
#
# _root = dirname(realpath(__file__))
# _root = join(_root, 'data', 'def')
#
# def test_create_data():
#
#     data = lim.create_data()
#
#     Y0 = lim.reader.csv(join(_root, 'expr_cond0.csv'), float, row_header=True, col_header=True)
#     Y0.set_axis_name(0, 'sample_id')
#     Y0.set_axis_name(1, 'gene')
#     Y0.indexed_axis = 0
#
#     Y1 = lim.reader.csv(join(_root, 'expr_cond1.csv'), float, row_header=True, col_header=True)
#     Y1.set_axis_name(0, 'sample_id')
#     Y1.set_axis_name(1, 'gene')
#     Y1.indexed_axis = 0
#
#     Y01 = lim.reader.group([Y0, Y1], name='condition', values=[0, 1])
#
#     Y2 = lim.reader.csv(join(_root, 'expr_cond2.csv'), float, row_header=True, col_header=True)
#     Y2.set_axis_name(0, 'gene')
#     Y2.set_axis_name(1, 'sample_id')
#     Y2.indexed_axis = 1
#
#     Y3 = lim.reader.csv(join(_root, 'expr_cond3.csv'), float, row_header=True, col_header=True)
#     Y3.set_axis_name(0, 'gene')
#     Y3.set_axis_name(1, 'sample_id')
#     Y3.indexed_axis = 1
#
#     Y23 = lim.reader.group([Y2, Y3], name='condition', values=[0, 1])
#
#     Y = lim.reader.group([Y01, Y23], name='planet', values=['mars', 'venus'])
#
#     data.add_sample_attrs(Y)
#
#     y = data.sample.planet('mars').condition(0).gene("geneA")
#     assert_array_equal(y, array([2.1, -3.1, 0.2, 0.13]))
#
#     yi = data.sample.planet('mars').condition(0).sample_id("sample_02")
#     assert_array_equal(yi, array([-3.1, 33.2, -23.]))
#
#     y = lim.reader.h5(join(_root, 'gene_expr.h5'), '/group/smoke_trait')
#
#     y.set_axis_name(0, 'sample_id')
#     y.set_axis_values(0, Y0.get_axis_values(0))
#     y.set_axis_name(1, 'env')
#     y.set_axis_values(1, 'smoke')
#
#     data.add_sample_attrs(y, name='phase2')
#     r = array(['y', 'n', 'n', 'y'])
#     assert_array_equal(data.sample.phase2.env('smoke'), r)
#
#     y = lim.reader.h5(join(_root, 'gene_expr.h5'), '/group/poor')
#
#     y.set_axis_name(0, 'sample_id')
#     y.set_axis_values(0, Y0.get_axis_values(0))
#     y.set_axis_name(1, 'env')
#     y.set_axis_values(1, 'poor')
#
#     data.add_sample_attrs(y, name='phase2')
#     y = data.sample.phase2.env('poor')
#     assert_array_almost_equal(y, array([ 0.55830158954378217,
#                                         -0.48993065101082273,
#                                         -1.4155466852892591,
#                                         -0.79204549998611107]))
#
#     ped = lim.reader.ped(join(_root, 'plink'))
#     # for (i, sid) in enumerate(ped.get_sample_ids()):
#     #     ped.set_sample_id(sid, Y0.get_axis_values(0)[i])
#     # (samples, markers, G) = lim.reader.ped(join(_root, 'plink'))
#     # data.add_sample_attrs(samples, 'ped')
#     # R = array([['1'], ['2'], ['3'], ['1'], ['2'], ['3']])
#     # assert_array_equal(data.sample.ped.attr('individual_id'), R)
#     #
#     # data.add_marker_attrs(markers, 'ped')
#     # r = array([1, 2, 3])
#     # assert_array_equal(data.marker.ped.attr('base_pair_position'), r)
#
#     # idx = data.sample.planet('mars').condition(0).gene("geneA") == data.sample.planet('mars').condition(0).gene("geneB")
