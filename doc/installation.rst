Installation
------------

For those new to the scientific Python community, we strongly recommend first
getting to know about `Anaconda <https://www.continuum.io/downloads>`_
platform.
In particular, we will refer to its package manager
`Conda <http://conda.pydata.org/docs/intro.html>`_ in this documentation.

We also recommend that the `Numba <http://numba.pydata.org>`_
package is installed (and up-to-date) beforehand.
This can be easily accomplished if you have Conda:

.. code-block:: console

    conda install numba

Finally, Lim can be installed via

.. code-block:: console

    pip install lim

The above command should install the latest Lim version. If that doesn't
happen, try instead

.. code-block:: console

    pip install lim --no-cache-dir

to prevent the use of a cached version in your system. And if you already have
Lim previously installed, you can upgrade it via

.. code-block:: console

    pip install lim --upgrade

In any case, make sure you have the latest version

.. code-block:: console

    python -c "import lim; print('Lim ' + lim.__version__)"

.. program-output:: python -c "import lim; print('Lim ' + lim.__version__)"

and that it is actually working

.. code-block:: console

    python -c "import lim; lim.test()"

.. program-output:: python -c "import lim; lim.test()"

Add-ons
^^^^^^^

Lim is also able to analyse count phenotype via a
Binomial likelihood. For that you need to install
an extra package

.. code-block:: console

    pip install git+https://github.com/Horta/limix-qep.git

which is only possible if you have given permission (contact horta@ebi.ac.uk).
