# lim

[![PyPIl](https://img.shields.io/pypi/l/lim.svg?style=flat-square)](https://pypi.python.org/pypi/lim/)
[![PyPIv](https://img.shields.io/pypi/v/lim.svg?style=flat-square)](https://pypi.python.org/pypi/lim/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/lim/badges/version.svg)](https://anaconda.org/conda-forge/lim)
[![Documentation Status](https://readthedocs.org/projects/lim/badge/?style=flat-square&version=latest)](http://lim.readthedocs.io/en/latest/?badge=latest)

Lim.

## Install

The recommended way of installing **will be** via
[conda](http://conda.pydata.org/docs/index.html)
```bash
conda install -c conda-forge lim
```

As of now, first make sure you have [conda-forge](https://conda-forge.github.io/)
channel set
```bash
conda config --add channels new_channel
```
and the [Numba](http://numba.pydata.org/) package installed::
```bash
conda install numba
```

Then type
```bash
pip install lim
```

## Running the tests

After installation, you can test it
```
python -c "import lim; lim.test()"
```
as long as you have [pytest](http://docs.pytest.org/en/latest/).

## Authors

* **Danilo Horta** - [https://github.com/Horta](https://github.com/Horta)

## License

This project is licensed under the MIT License - see the
[LICENSE](LICENSE) file for details
