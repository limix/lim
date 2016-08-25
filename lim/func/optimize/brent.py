import sys

from numpy import asarray

from scipy.optimize import brent

from ..negative import negative_function


def minimize(function, verbose=False):

    def func(x):
        if verbose:
            sys.stdout.write('.')
        x = asarray(x).ravel()
        function.variables().from_flat(x)
        return function.value()

    x = brent(func)
    if verbose:
        print("")
    function.variables().from_flat(asarray(x).ravel())


def maximize(function, verbose=False):
    function = negative_function(function)
    return minimize(function, verbose=verbose)
