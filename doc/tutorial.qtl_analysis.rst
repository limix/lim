QTL Analysis
============

The first example uses :func:`lim.genetics.qtl.normal_scan` to perform an
association scan between markers contained in `X` and the phenotype defined by
`y`, while accounting for background signal via `G`:

.. literalinclude:: /../examples/normal_qtl_analysis.py

The output should be similar to:

.. program-output:: python ../examples/normal_qtl_analysis.py
