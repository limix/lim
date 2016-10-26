from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

import logging

from glob import glob
from os.path import join
import six

def make_sure_string(msg):
    if six.PY2:
        return bytes(msg)
    else:
        return u"%s" % __builtins__['str'](msg)

def _make():
    from cffi import FFI
    from pycflags import get_c11_flag

    logger = logging.getLogger()

    logger.debug('CFFI make')

    ffi = FFI()


    rfolder = join(b'lim', b'inference', b'ep', b'liknorm', b'clib')

    sources = glob(join(rfolder, b'liknorm', b'*.c'))
    sources += [join(rfolder, b'liknorm.c')]
    sources = [make_sure_string(s) for s in sources]

    hdrs = glob(join(rfolder, b'liknorm', b'*.h'))
    hdrs += [join(rfolder, b'liknorm.h')]
    hdrs = [make_sure_string(h) for h in hdrs]

    incls = [join(rfolder, b'liknorm')]
    incls = [make_sure_string(i) for i in incls]
    libraries = ['m']

    logger.debug("Sources: %s", str(sources))
    logger.debug('Headers: %s', str(hdrs))
    logger.debug('Incls: %s', str(incls))
    logger.debug('Libraries: %s', str(libraries))

    ffi.set_source('lim.inference.ep.liknorm._liknorm_ffi',
                   '''#include "liknorm.h"''',
                   include_dirs=incls,
                   sources=sources,
                   libraries=libraries,
                   library_dirs=[],
                   depends=sources + hdrs,
                   extra_compile_args=[get_c11_flag()])

    with open(join(rfolder, b'liknorm.h'), 'r') as f:
        ffi.cdef(f.read())

    return ffi

liknorm = _make()
