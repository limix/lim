class ReaderGroup(object):
    def __init__(self, items, name, values=None):
        if values is None:
            self._items = items[0]
        else:
            self._items = {values[i]:items[i] for i in range(len(values))}
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, *args):
        if len(args) == 0:
            return self._items
        return self._items[args[0]]

    def __getattr__(self, args):
        if args == self._name:
            return self
        return getattr(self._items, args)

def group(items, name, values=None):
    return ReaderGroup(items, name, values=values)
