from __future__ import division, print_function
import os
import sys
from setuptools import setup, find_packages


def make_sure_install(package):
    import pip
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package, '--upgrade'])
make_sure_install('build_capi')
make_sure_install('ncephes')


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    setup_requires = ['cffi>=1.6', 'pytest-runner', 'build_capi>=0.0.4',
                      'ncephes>=0.0.29']
    install_requires = ['limix_math>=0.2.4', 'cffi>=1.6', 'bidict',
                        'pytest', 'numpy>=1.9', 'scipy>=0.17', 'numba>=0.27',
                        'ncephes>=0.0.29', 'tabulate>=0.7', 'pandas>=0.18',
                        'h5py>=2.6', 'progressbar>=3.10']
    tests_require = ['pytest', 'tabulate>=0.7', 'pandas>=0.18', 'h5py>=2.6',
                     'progressbar>=3.10']

    metadata = dict(
        name='lim',
        version='0.0.8',
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

    try:
        from distutils.command.bdist_conda import CondaDistribution
    except ImportError:
        pass
    else:
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
