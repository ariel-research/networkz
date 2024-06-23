"""

The Paper - 
Approximability of the Firefighter Problem Computing Cuts over Time

Paper Link -
https://www.math.uwaterloo.ca/~cswamy/papers/firefighter-journ.pdf

Authors - 
Elliot Anshelevich
Deeparnab Chakrabarty
Ameya Hate 
Chaitanya Swamy

Developers - 
Yuval Bubnovsky
Almog David
Shaked Levi

"""

import networkx as nx
import pytest
import sys
import os

from networkz.algorithms.approximation.firefighter_problem.Utils import *
## TODO --> This is not implementaed for now, all the methods are being tested in their corresponding usage in other tests.py.
## we might consider changing it later for a better coding 

@pytest.fixture
def test_graph():
    """
    Create a pre-determined graph for testing, minimizes code duplication in tests - this is called at the beginning of each test
    """
    pass

@pytest.fixture
def test_dirlay():
    """
    Create a pre-determined directed layered network for testing.
    """
    pass

def test_validate_parameters(test_graph):
    pass

def test_spread_virus(test_graph):
    pass

def test_spread_vaccination(test_graph):
    pass

def test_vaccinate_node(test_graph):
    pass

def test_clean_graph(test_graph):
    pass

def test_adjust_nodes_capacity(test_graph):
    pass

def test_create_st_graph(test_graph):
    pass

def test_flow_reduction(test_graph):
    pass

def test_calculate_vaccine_matrix(test_dirlay):
    pass

def test_display_graph(test_graph):
    pass