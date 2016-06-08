class ReaderGroup(object):
    def __init__(self, items, name, values):
        self._items = items
        self._name = name
        self._values = values

def group(items, name, values):
    return ReaderGroup(items, name, values)
