from numpy import asarray

class Index(object):
    def __init__(self, indices):
        self._indices = indices

def index_array(dict_like, names):
    return asarray([dict_like[n] for n in names])
