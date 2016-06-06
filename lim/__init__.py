def define_data():
    from .data.data import Data
    return Data()

def h5path(filepath, itempath, dtype=None):
    from .data.h5 import H5Path
    return H5Path(filepath, itempath, dtype=dtype)

def csvpath(filepath, dtype=float):
    from .data.csv import CSVPath
    return CSVPath(filepath, dtype=dtype)

def vcfpath(filepath):
    from .data.vcf import VCFPath
