from __future__ import absolute_import as _absolute_import

import logging as _log
from logging.config import fileConfig as _fileConfig

try:
    _fileConfig('config.ini')
except Exception:
    _log.basicConfig(level=_log.INFO)

from . import genetics
from . import random
from . import tool
from . import util

from pkg_resources import get_distribution as _get_distribution
from pkg_resources import DistributionNotFound as _DistributionNotFound

try:
    __version__ = _get_distribution('lim').version
except _DistributionNotFound:
    __version__ = 'unknown'


def test():
    import os
    p = __import__('lim').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main(['-q'])
    finally:
        os.chdir(old_path)

    if return_code == 0:
        print("Congratulations. All tests have passed!")

    return return_code
