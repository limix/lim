import inspect

class UniFuncDataAttacher(object):
    def __init__(self, target, data):
        self._target = target
        self._data = data

    def value(self):
        return self._target.value(self._data)

    def gradient(self):
        return self._target.gradient(self._data)

    @property
    def data(self):
        return self._data

class BiFuncDataAttacher(object):
    def __init__(self, target, data_left, data_right):
        self._target = target
        self._data_left = data_left
        self._data_right = data_right

    def value(self):
        return self._target.value(self._data_left, self._data_right)

    def gradient(self):
        return self._target.gradient(self._data_left, self._data_right)

    @property
    def data_left(self):
        return self._data_left

    @property
    def data_right(self):
        return self._data_right

def set_data(obj, data_left, data_right=None, purpose='learn'):
    nargs = len(inspect.getargspec(obj.value).args) - 1
    if nargs == 1:
        fw = UniFuncDataAttacher(obj, data_left)
    elif nargs == 2:
        if data_right is None:
            data_right = data_left
        fw = BiFuncDataAttacher(obj, data_left, data_right)
    else:
        assert False
    setattr(obj, purpose, fw)

def unset_data(obj, purpose='learn'):
    delattr(obj, purpose)
