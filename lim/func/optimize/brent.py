import numpy as np
from scipy.optimize import brent

from ..negative import negative_function

def minimize(function):

    def func(x):
        x = np.asarray(x).ravel()
        function.variables().from_flat(x)
        return function.value()

    x = brent(func)
    function.variables().from_flat(np.asarray(x).ravel())

def maximize(function):
    function = negative_function(function)
    return minimize(function)
