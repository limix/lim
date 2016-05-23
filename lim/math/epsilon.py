from numpy import finfo
from numpy import sqrt

class Epsilon(object):
    __slots__ = ['tiny', 'small', 'large']
    def __init__(self):
        self.tiny = finfo(float).eps
        self.small = sqrt(finfo(float).eps)
        self.large = sqrt(sqrt(finfo(float).eps))

epsilon = Epsilon()
