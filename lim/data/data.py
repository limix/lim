import sqlite3

def _py_type_to_sql_type(type_):
    if type_ is str or type_ is bytes:
        return 'text'
    if type_ is int:
        return 'integer'
    if type_ is float:
        return 'real'
    raise TypeError('Unknown data type.')

def _create_sqlfields(attrs):
    fields = '( '
    for (name, attr) in iter(attrs.items()):
        ftype = _py_type_to_sql_type(attr.dtype)
        fields += '%s %s, ' % (name, ftype)
    fields = fields[:-2] + ' )'

    return fields

def _insert_attributes(cursor, table_name, attrs):
    attrs = [v[:] for v in attrs.values()]
    nattrs = len(attrs)
    nvalues = len(attrs[0])
    for i in range(nvalues):
        sql  = "INSERT INTO trait_%s_sample VALUES " % (table_name,)
        sql += "(" + ','.join(['?']*nattrs) + ")"
        cursor.execute(sql, [attr[i] for attr in attrs])

class Data(object):
    def __init__(self):
        self._conn = sqlite3.connect(':memory:')
        self._trait = dict()
        self._genotype = dict()

    def add_trait(self, path, id, sample_attrs):
        self._trait[id] = path
        table_name = "trait_%s_sample" % (id,)

        c = self._conn.cursor()
        fields = _create_sqlfields(sample_attrs)
        c.execute("CREATE TABLE trait_%s_sample %s" % (table_name, fields))

        _insert_attributes(c, table_name, sample_attrs)

        self._conn.commit()

    def add_genotype(self, path, id, sample_attrs, marker_attrs):
        self._genotype[id] = path

        c = self._conn.cursor()
        import ipdb; ipdb.set_trace()

        fields = _create_sqlfields(sample_attrs)
        table_name = "genotype_%s_sample" % (id,)
        c.execute("CREATE TABLE %s %s" % (table_name, fields))
        _insert_attributes(c, table_name, sample_attrs)

        fields = _create_sqlfields(marker_attrs)
        table_name = "genotype_%s_marker" % (id,)
        c.execute("CREATE TABLE genotype_%s_marker %s" % (table_name, fields))
        _insert_attributes(c, table_name, marker_attrs)

        self._conn.commit()

    def select(self, traits, genotypes):
        pass
