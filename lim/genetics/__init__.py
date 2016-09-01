"""Genetic related analysis.

The provided analysis are:
    - Heritability estimation;
    - Quantitative trait locus (QTL) discovery.
"""

from .core import FastLMM
from .transformation import DesignMatrixTrans
from . import heritability
from . import qtl
from .model import NormalModel
from .model import BinomialModel
