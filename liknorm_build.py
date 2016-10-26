from __future__ import (division, absolute_import, unicode_literals,
                        print_function)

import logging

from glob import glob
from os.path import join


def _make():
    from cffi import FFI

    logger = logging.getLogger()

    logger.debug('CFFI make')

    ffi = FFI()

    rfolder = join('lim', 'inference', 'ep', 'liknorm', 'clib')

    sources = glob(join(rfolder, 'liknorm', '*.c'))
    sources += [join(rfolder, 'liknorm.c')]

    hdrs = glob(join(rfolder, 'liknorm', '*.h'))
    hdrs += [join(rfolder, 'liknorm.h')]

    incls = [join(rfolder, 'liknorm')]
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
                   extra_compile_args=["-std=c11"])

    with open(join(rfolder, 'liknorm.h'), 'r') as f:
        ffi.cdef(f.read())

    return ffi

liknorm = _make()
