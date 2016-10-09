from __future__ import absolute_import
from . import cov
from . import data
from . import func
from . import genetics
from . import inference
from . import math
from . import mean
from . import random
from . import reader
from . import tool
from . import util
from pkg_resources import get_distribution

__version__ = get_distribution('lim').version


def test():
    import os
    p = __import__('lim').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main([])
    finally:
        os.chdir(old_path)

    return return_code

__all__ = ['genetics', 'reader']
