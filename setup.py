from __future__ import division, print_function
import os
import sys
from setuptools import setup, find_packages

PKG_NAME = 'lim'
VERSION  = '0.0.1'

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

try:
    import scipy
except ImportError:
    print("Error: scipy package couldn't be found."+
          " Please, install it so I can proceed.")
    sys.exit(1)

try:
    import numba
except ImportError:
    print("Error: numba package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)

def write_version():
    cnt = """
# THIS FILE IS GENERATED FROM %(package_name)s SETUP.PY
version = '%(version)s'
"""
    filename = os.path.join(PKG_NAME, 'version.py')
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'package_name': PKG_NAME.upper()})
    finally:
        a.close()

def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    write_version()

    install_requires = ['limix_math']
    setup_requires = []

    metadata = dict(
        name=PKG_NAME,
        version=VERSION,
        maintainer="Limix Developers",
        maintainer_email = "horta@ebi.ac.uk",
        license="BSD",
        url='http://pmbio.github.io/limix/',
        test_suite='setup.get_test_suite',
        packages=find_packages(),
        zip_safe=True,
        install_requires=install_requires,
        setup_requires=setup_requires
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
