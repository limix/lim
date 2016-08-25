from __future__ import absolute_import
from pkg_resources import get_distribution

__version__ = get_distribution('lim').version

# from . import reader
# from .data import create_data
from . import genetics


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
