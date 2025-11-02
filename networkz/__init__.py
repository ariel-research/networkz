from networkx import *
import os

__version__ = "1.0.6"

# Get the current directory of the __init__.py file
current_directory = os.path.dirname(__file__)

# Append the 'algorithms' directory to the __path__ attribute
__path__.append(os.path.join(current_directory, 'algorithms'))

from networkz import algorithms
from networkz.algorithms import *
