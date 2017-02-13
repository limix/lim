# lim

[![PyPI-License](https://img.shields.io/pypi/l/lim.svg?style=flat-square)](https://pypi.python.org/pypi/lim/)
[![PyPI-Version](https://img.shields.io/pypi/v/lim.svg?style=flat-square)](https://pypi.python.org/pypi/lim/)
[![Documentation Status](https://readthedocs.org/projects/lim/badge/?style=flat-square&version=latest)](http://lim.readthedocs.io/en/latest/)

Lim is an efficient implementation of Generalized Linear Mixed Models for
genomic analysis.

## Install

The recommended way of installing it is via [conda](http://conda.pydata.org/docs/index.html)

```bash
conda install -c conda-forge limix-inference
conda install h5py pandas tabulate pytest
```

and then

```bash
pip install lim
```

## Running the tests

After installation, you can test it
```
python -c "import lim; lim.test()"
```
as long as you have [pytest](http://docs.pytest.org/en/latest/).

## Documentation

Refer to the [documentation](http://lim.readthedocs.io/en/latest/) for detailed
information.

## Authors

* **Christoph Lippert** - [https://github.com/clippert](https://github.com/clippert)
* **Danilo Horta** - [https://github.com/Horta](https://github.com/Horta)
* **Oliver Stegle** - [https://github.com/ostegle](https://github.com/ostegle)
* **Paolo Francesco Casale** - [https://github.com/fpcasale](https://github.com/fpcasale)

## License

This project is licensed under the MIT License -- see the
[LICENSE](LICENSE) file for details.
