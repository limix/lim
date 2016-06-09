from os.path import join
from os.path import dirname
from os.path import realpath

from numpy import array
from numpy.testing import assert_array_equal

import lim

_root = dirname(realpath(__file__))
_root = join(_root, 'data', 'def')

def test_create_data():

    data = lim.create_data()

    Y0 = lim.reader.csv(join(_root, 'expr_cond0.csv'), float, row_header=True, col_header=True)
    Y0.set_axis_name(0, 'sample_id')
    Y0.set_axis_name(1, 'gene')

    Y1 = lim.reader.csv(join(_root, 'expr_cond1.csv'), float, row_header=True, col_header=True)
    Y1.set_axis_name(0, 'sample_id')
    Y1.set_axis_name(1, 'gene')

    Y01 = lim.reader.group([Y0, Y1], name='condition', values=[0, 1])

    Y2 = lim.reader.csv(join(_root, 'expr_cond2.csv'), float, row_header=True, col_header=True)
    Y2.set_axis_name(0, 'gene')
    Y2.set_axis_name(1, 'sample_id')

    Y3 = lim.reader.csv(join(_root, 'expr_cond3.csv'), float, row_header=True, col_header=True)
    Y3.set_axis_name(0, 'gene')
    Y3.set_axis_name(1, 'sample_id')

    Y23 = lim.reader.group([Y2, Y3], name='condition', values=[0, 1])

    Y = lim.reader.group([Y01, Y23], name='planet', values=['mars', 'venus'])

    data.add_sample_attrs(Y)

    y = data.planet('mars').condition(0).gene("geneA")
    assert_array_equal(y, array([2.1, -3.1, 0.2, 0.13]))

    yi = data.planet('mars').condition(0).sample_id("sample_02")
    assert_array_equal(yi, array([-3.1, 33.2, -23.]))

    y = lim.reader.h5(join(_root, 'gene_expr.h5'), '/group/smoke_trait')
    y.set_axis_name(0, 'sample_id')
    y.set_axis_values(0, Y0.get_axis_values(0))

    y.set_axis_name(1, 'env')
    y.set_axis_values(1, 'smoke')

    data.add_sample_attrs(y, name='phase2')
    assert_array_equal(data.phase2.env('smoke'), array(['y', 'n', 'n', 'y']))

    # y = lim.h5_reader('gene_expr.h5', '/group/height')
    # y.set_axis_name(0, 'sample_id')
    # y.set_axis_values(0, Y0.get_axis_values(0))
    # data.add_sample_attrs(y)
    #
    # (Y, M, G) = lim.bed_reader('base_name')
    #
    # data.add_sample_attrs(Y)
    # data.add_marker_attrs(M)
    # data.add_genotype(G)
    #
    # geneA = data.sample.condition(0).planet('mars').geneA
    # geneB = data.sample.condition(0).planet('venus').geneB
    # data = data.subsample[geneA == geneB]
    #
    # bp_pos = data.marker.chrom('1').bp_pos
    # data = data.submarker[bp_pos > 1000]
    #
    # y = data.sample.condition(0).planet('mars').geneB.value
    # G = data.genotype.chrom('1').value
    #
    # print(y)
    # print(G)

#
#
#
# Y2.set_row_index(sample_id=Y0.get_row_index('sample_id'))
# Y2.set_col_index(name=Y0.get_col_index('trait_name'), condition=2)
#
#
#
# Y0.set_col_index(condition=0)
#
# Y1 = lim.csv_reader('expr_cond1.csv', row_header=True, row_header_name='sample_id',
#                     col_header=True, col_header_name='trait_name', dtype=float)
# Y1.set_col_index(condition=1)
#
#
#
# data.add_sample_attrs(Y0)
# data.add_sample_attrs(Y1)
# data.add_sample_attrs(Y2)
#
# (Y, GA, G) = lim.bed_reader('base_name0')
#
# data.add_sample_attrs(Y)
# data.add_marker_attrs(GA)
# data.add_genotype(G)
#
# (Y, GA, G) = lim.ped_reader('base_name1')
#
# data.add_sample_attrs(Y)
# data.add_marker_attrs(GA)
# data.add_genotype(G)
#
# ###########################################################################
# # GRM (Genetic Realized Matrix reader for plink)
# # Output formats: 'square', 'square0', 'triangle', 'gz', 'bin', and 'bin4'
# # Extensions: .rel, .rel.bin, or .rel.gz
# ###########################################################################
# K = lim.gcta_grm_reader('grm.rel')
# K.set_row_index('sample_id', Y0.get_row_index('sample_id'))
# K.set_col_index('sample_id', Y0.get_row_index('sample_id'))
#
# data.add_covariance(K)
#
#
# # SUBSELECTION
# ##############
#
# data = data[data.trait['awsome_gene1'] > 0]
# data.sample_select('awsome_gene > ')
# ds = d.subsample('awsome_gene>5')
# y = ds.get_pheno(pheno_id='shitty_gene')
#
#
#
#
#
#
#
#
#
#
#
#
# Y = lim.csvpath('expr.csv')
#
# sample_ids = Y[0,1:]
# trait_ids = Y[1:,0]
#
# d = limix.create_data()
#
# for i in range(1, Y.shape[0]):
#   d.add_trait(trait_ids[i], Y[i, 1:], sample_ids[i])
#
# G = limix.csvpath('annotation.csv')
#
# trait_ids = G[1:,0]
# trait_attr_names = G[0,1:]
#
# for i in range(len(trait_ids)):
#   data.add_trait_attr(trait_ids[i], trait_attr_names[i], G[i,1:])
#
#
# ################################################
# # Example of univariate eQTL mapping
# ################################################
# Y = lim.csvpath('expr.csv', col_header=True, row_header=True) # default should be True
# G = limix.csvpath('annotation.csv', col_header=True, row_header=True) # default should be True
#
# d = limix.create_data()
# d.add_pheno(Y, attributes=G, samples='cols') # attribute=None, samples='cols'
# d.get_pheno(gene_chrom=1)
#
# # how about I want do subselect the phenotype based on other phenotypes
# # I want to get the expression of shitty_gene when awsome_gene>5
# ds = d.subsample('awsome_gene>5')
# y = ds.get_pheno(pheno_id='shitty_gene')
#
# ################################################
# # Example of multi-variate eQTL mapping
# # Is this a good solution?
# ################################################
# Y1 = lim.csvpath('expr_cond1.csv', col_header=True, row_header=True) # default should be True
# Y2 = lim.csvpath('expr_cond2.csv', col_header=True, row_header=True) # default should be True
# Yg = lim.csvpath('global_phenos.csv', col_header=True, row_header=True)
# Ym = lim.csvpath('meta.csv', col_header=True, row_header=True) # 'male', 'female'
# G = limix.csvpath('annotation.csv', col_header=True, row_header=True) # default should be True
#
# d = limix.create_data()
# d.add_pheno(Y1, attributes=G,, samples='cols', condition='cond1') # attribute=None, samples='cols'
# d.add_pheno(Y2, attributes=G, samples='cols', condition='cond1) # attribute=None, samples='cols' # attribute=None, samples='cols'
# d.add_pheno(Yg, samples='cols')
# d.add_pheno(Ym, samples='cols')
#
# str_vec = d.get_pheno(trait_id='gender')
# # str_vec in a #samplesx2
#
# y.get_pheno
