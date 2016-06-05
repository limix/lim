class VCFPath(object):
    def __init__(self, filepath, row=None, col=None):
        self._filepath = filepath
        self._row = row
        self._col = col

    def read_array(self):
        pass
