Installation
------------

You can install via

.. code-block:: console

    pip install lim

The above command should install the latest Lim version. If that doesn't
happen, try instead

.. code-block:: console

    pip install lim --no-cache-dir

to prevent cache. And if you already have Lim previously installed, you
can upgrade it via

.. code-block:: console

    pip install lim --upgrade

In any case, make sure you have the latest version

.. code-block:: console

    python -c "import lim; print(lim.__version__)"

and that it is actually working

.. code-block:: console

    python -c "import lim; lim.test()"
