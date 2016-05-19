import inspect

class UniFuncDataWrapper(object):
    def __init__(self, target, data):
        self._target = target
        self.raw = [data]

    def value(self):
        return self._target.value(self.raw[0])

    def gradient(self):
        return self._target.gradient(self.raw[0])

class UniFuncData(object):
    def __init__(self):
        self.__data = dict()

    def data(self, purpose='learn'):
        return UniFuncDataWrapper(self, self.__data[purpose])

    def set_data(self, data, purpose='learn'):
        self.__data[purpose] = data

    def unset_data(self, purpose='learn'):
        del self.__data[purpose]

class BiFuncDataWrapper(object):
    def __init__(self, target, data_left, data_right):
        self._target = target
        self.raw = [data_left, data_right]

    def value(self):
        return self._target.value(self.raw[0], self.raw[1])

    def gradient(self):
        return self._target.gradient(self.raw[0], self.raw[1])

class BiFuncData(object):
    def __init__(self):
        self.__data = dict()

    def data(self, purpose='learn'):
        return BiFuncDataWrapper(self, self.__data[purpose][0],
                                 self.__data[purpose][1])

    def set_data(self, data_left, data_right, purpose='learn'):
        self.__data[purpose] = [data_left, data_right]

    def unset_data(self, purpose='learn'):
        del self.__data[purpose]
