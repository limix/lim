import os
from os.path import join

import lim

def test_data_definition():
    root = os.path.dirname(os.path.realpath(__file__))
    root = join(root, 'data', 'def')

    d = lim.define_data()

    smoke = lim.csvpath(join(root, 'smoke.csv'), bytes)
    d.add_sample_attrs('smoke', smoke)

    X, sample_attrs, marker_attrs = lim.plinkpaths(join(root, 'example'))
    for (id_, attr) in iter(marker_attrs.items()):
        d.add_sample_attrs(id_, attr)

    d.add_genotype('geno', X)

    for (id_, attr) in iter(marker_attrs.items()):
        d.add_marker_attrs(id_, attr)

    d.add_trait('trait', lim.h5path(join(root, 'example.h5'), '/group/trait'))

    dsmokers = d.where("smoke = 'y'")
    dchrom1 = d.where("chrom = 1")

    print("")
    print(d)
    print("")
    print(dsmokers)
    print("")
    print(dchrom1)

    # d.add_trait(lim.h5path(join(root, 'example.h5'), '/group/trait'), 'trait')
    # # d.add_genotype(X, 'marker')
    #
    # d = d.where('trait.sample_id > 0')


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
