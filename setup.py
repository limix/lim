from __future__ import division, print_function
import os
import sys
from setuptools import setup, find_packages

PKG_NAME = 'lim'
VERSION = '0.0.6.dev3'

try:
    from distutils.command.bdist_conda import CondaDistribution
except ImportError:
    conda_present = False
else:
    conda_present = True

try:
    import numpy
except ImportError:
    print("Error: numpy package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)
else:
    print("Good: numpy %s" % numpy.__version__)

try:
    import scipy
except ImportError:
    print("Error: scipy package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)
else:
    print("Good: numpy %s" % scipy.__version__)

try:
    import numba
except ImportError:
    print("Error: numba package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)
else:
    print("Good: numba %s" % numba.__version__)


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    install_requires = ['limix_math>=0.2.1', 'cffi>=1.0.0', 'bidict',
                        'pytest']
    setup_requires = ['cffi>=1.0.0', 'pytest-runner']
    tests_require = ['pytest']

    metadata = dict(
        name=PKG_NAME,
        version=VERSION,
        maintainer="Limix Developers",
        maintainer_email="horta@ebi.ac.uk",
        license="BSD",
        url='http://pmbio.github.io/limix/',
        packages=find_packages(),
        zip_safe=True,
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        cffi_modules=["lim/reader/cplink/bed.py:ffi"],
    )

    if conda_present:
        metadata['distclass'] = CondaDistribution
        metadata['conda_buildnum'] = 1
        metadata['conda_features'] = ['mkl']

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

if __name__ == '__main__':
    setup_package()
