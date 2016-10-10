from __future__ import division

from numpy import log
from numpy import exp


class Link(object):

    def __init__(self):
        super(Link, self).__init__()

    def value(self, x):
        raise NotImplementedError

    def inv(self, x):
        raise NotImplementedError


class Logit(Link):

    def __init__(self):
        super(Logit, self).__init__()

    def value(self, x):
        return log(x / (1 - x))

    def inv(self, x):
        return 1 / (1 + exp(-x))


class Log(Link):

    def __init__(self):
        super(Log, self).__init__()

    def value(self, x):
        return log(x)

    def inv(self, x):
        return exp(x)
