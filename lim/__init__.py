from pkg_resources import get_distribution

__version__ = get_distribution('lim').version

# from . import reader
# from .data import create_data
from . import genetics
