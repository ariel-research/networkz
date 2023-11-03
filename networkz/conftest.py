"""
Testing
=======

General guidelines for writing good tests:

- doctests always assume ``import networkx as nx`` so don't add that
- prefer pytest fixtures over classes with setup methods.
- use the ``@pytest.mark.parametrize``  decorator
- use ``pytest.importorskip`` for numpy, scipy, pandas, and matplotlib b/c of PyPy.
  and add the module to the relevant entries below.

"""

# What dependencies are installed?

try:
    import numpy

    has_numpy = True
except ImportError:
    has_numpy = False

try:
    import scipy

    has_scipy = True
except ImportError:
    has_scipy = False



# List of files that pytest should ignore

collect_ignore = []

needs_numpy = [
    "algorithms/max_weight_fractional_matching.py",
]
needs_scipy = [
    "algorithms/max_weight_fractional_matching.py",
]

if not has_numpy:
    collect_ignore += needs_numpy
if not has_scipy:
    collect_ignore += needs_scipy
