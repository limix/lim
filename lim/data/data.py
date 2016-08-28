import sqlite3

from pandas import DataFrame
from pandas import read_sql

from numpy import asarray
from numpy import arange

from ..reader.group import group


def _make_sure_ids(ids, n):
    if ids is None:
        ids = arange(n, dtype=int)
    return ids


def create_data():
    return Data()

# class DataView(object):
#     def __init__(self, ref, sample_attrs, marker_attrs):
#         self._ref = ref
#         self._sample_attrs = sample_attrs
#         self._marker_attrs = marker_attrs
#
#     def genotype(self, genotype_id):
#         pass
#
#     @property
#     def sample_attrs(self):
#         return self._sample_attrs
#
#     @property
#     def marker_attrs(self):
#         return self._marker_attrs
#
#     def __repr__(self):
#         return repr(self._sample_attrs) + '\n' + repr(self._marker_attrs)
#
#     def __str_(self):
#         return bytes(self._sample_attrs) + '\n' + bytes(self._marker_attrs)


class DictAccessor(object):

    def __init__(self, dict_):
        self._dict = dict_

    def __getattr__(self, attr):
        if attr in self._dict:
            return self._dict[attr]
        raise AttributeError


class Data(object):

    def __init__(self):
        self._sample_groups = dict()
        self._marker_groups = dict()

    def add_marker_attrs(self, M, name=None):
        if name is None:
            self._marker_groups[M.name] = M
        else:
            self._marker_groups[name] = group([M], name=name)

    def add_sample_attrs(self, Y, name=None):
        if name is None:
            self._sample_groups[Y.name] = Y
        else:
            self._sample_groups[name] = group([Y], name=name)

        # sample_ids = _make_sure_ids(sample_ids, len(attrs))
        #
        # for (i, a) in enumerate(attrs):
        #     self._sample_attrs.set_value(sample_ids[i], attr_id, a)

    @property
    def sample(self):
        return DictAccessor(self._sample_groups)

    @property
    def marker(self):
        return DictAccessor(self._marker_groups)

    # def add_marker_attrs(self, attr_id, attrs, genotype_ids, marker_ids=None):
    #     marker_ids = _make_sure_ids(marker_ids, len(attrs))
    #
    #     for (i, a) in enumerate(attrs):
    #         self._marker_attrs.set_value(marker_ids[i], attr_id, a)
    #
    # def add_genotype(self, genotype_id, X, sample_ids=None, marker_ids=None):
    #     sample_ids = _make_sure_ids(sample_ids, X.shape[0])
    #     marker_ids = _make_sure_ids(marker_ids, X.shape[1])
    #
    #     if genotype_id not in self._genotypes:
    #         self._genotypes = []
    #
    #     self._genotypes[genotype_id].append((sample_ids, marker_ids, X))
    #
    # def add_trait(self, trait_id, Y, sample_ids=None):
    #     sample_ids = _make_sure_ids(sample_ids, Y.shape[0])
    #
    #     if trait_id not in self._traits:
    #         self._traits = []
    #
    #     self._traits[trait_id].append((sample_ids, Y))
    #
    # @property
    # def sample_attrs(self):
    #     return self._sample_attrs
    #
    # @property
    # def marker_attrs(self):
    #     return self._marker_attrs
    #
    # def where(self, sample_query='1 = 1', marker_query='1 = 1'):
    #
    #     conn = sqlite3.connect(':memory:')
    #
    #     self._sample_attrs.to_sql('sample_attrs', conn,
    #                               index_label='sample_id')
    #     self._marker_attrs.to_sql('marker_attrs', conn,
    #                               index_label='marker_id')
    #
    #     s = read_sql("SELECT * FROM sample_attrs WHERE %s" % sample_query,
    #                  conn, index_col="sample_id")
    #     m = read_sql("SELECT * FROM marker_attrs WHERE %s" % marker_query,
    #                  conn, index_col="marker_id")
    #
    #     conn.close()
    #
    #     return DataView(self, s, m)
    #
    # def __repr__(self):
    #     return repr(self._sample_attrs) + '\n' + repr(self._marker_attrs)
    #
    # def __str_(self):
    #     return bytes(self._sample_attrs) + '\n' + bytes(self._marker_attrs)
