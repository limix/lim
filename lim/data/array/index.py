class ArrayIdx(object):
    def __init__(self, index):
        self._index = index
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
