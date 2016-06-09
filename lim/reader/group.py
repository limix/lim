class ReaderGroup(object):
    def __init__(self, items, name, values):
        self._items = {values[i]:items[i] for i in range(len(values))}
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, args):
        return self._items[args]

    def __getattr__(self, args):
        if args == self._name:
            return self
        raise AttributeError

def group(items, name, values):
    return ReaderGroup(items, name, values)
