# import os
#
# import lim
#
# def test_data_definition():
#     root = os.path.dirname(os.path.realpath(__file__))
#     root = os.path.join(root, 'data', 'horta1')
#
#     d = lim.define_data()
#
#     trait_fp = os.path.join(root, 'trait.hdf5')
#     sample_attrs = dict(id=lim.h5path(trait_fp, '/group1/trait1/sample_id'))
#     d.add_trait(lim.h5path(trait_fp, '/group1/trait1/array'), 'trait1',
#                 sample_attrs=sample_attrs)
#
#     fp = os.path.join(root, 'genotype_array.csv')
#     sample_attrs_fp = os.path.join(root, 'genotype_sample_attrs.csv')
#     marker_attrs_fp = os.path.join(root, 'genotype_marker_attrs.csv')
#     sample_attrs = dict(id=lim.csvpath(sample_attrs_fp)[:,0])
#     marker_attrs = dict(id=lim.csvpath(marker_attrs_fp)[:,0],
#                         position=lim.csvpath(marker_attrs_fp)[:,1])
#     d.add_genotype(lim.csvpath(fp), 'genotype1', sample_attrs=sample_attrs,
#                    marker_attrs=marker_attrs)
#
#     d.select(traits=['trait1'], genotypes=['genotype1'])\
#      .where('trait1.id=genotype1.id')
#
# test_data_definition()
