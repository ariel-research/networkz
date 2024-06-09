"""

The Paper - 
Approximability of the Firefighter Problem Computing Cuts over Time

Paper Link -
https://github.com/The-Firefighters/networkz/blob/master/networkz/algorithms/approximation/firefighter_problem/Approximability_of_the_Firefighter_Problem.pdf

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

import pytest
import networkx as nx
import json
import numpy as np
import math

from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import non_spreading_dirlaynet_minbudget
from networkz.algorithms.approximation.firefighter_problem.Utils import adjust_nodes_capacity
from networkz.algorithms.approximation.firefighter_problem.Utils import create_st_graph
from networkz.algorithms.approximation.firefighter_problem.Utils import parse_json_to_networkx
from networkz.algorithms.approximation.firefighter_problem.Utils import graph_flow_reduction
from networkz.algorithms.approximation.firefighter_problem.Utils import calculate_vaccine_matrix
from networkz.algorithms.approximation.firefighter_problem.Utils import min_cut_N_groups
from networkz.algorithms.approximation.firefighter_problem.Utils import matrix_to_integers_values
from networkz.algorithms.approximation.firefighter_problem.Utils import min_budget_calculation

def get_graphs(): 
    with open("graphs.json", "r") as file:
        json_data = json.load(file)
    graphs = parse_json_to_networkx(json_data=json_data)
    return graphs

graphs = get_graphs() 

def test_source_not_in_graph():
    """
    This test checks if the source node is not a real node in the graph.
    """
    pattern = r"Error: The source node is( not|n't) on the graph"
    
    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-1"], -3, [0, 5])

    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-2"], 13, [0, 1, 4])
    
    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-3"], 15, [0, 6, 7])

    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-4"], -1, [1, 3, 5, 7])


def test_target_not_in_graph():
    """
    This test checks if a node we're trying to save is not in the graph.
    """
    pattern = r"Error: Not all nodes (we're trying to save|in the targets list) are on the graph"
    
    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-1"], 0, [1, 5, 7])

    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-2"], 1, [0, 2, -1, 9])
    
    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-3"], 4, [0, 1, 2, 11, 12, 13, 14])

    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-4"], 0, [1, 3, 5, 7, 15, 20])


def test_source_is_target():
    """
    This test checks if we're trying to save a source node.
    """
    pattern = r"Error: The source node can( not|'t) be a part of the targets (vector|list), since the virus is spreading from the source"
    
    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-1"], 0, [0, 5])

    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-2"], 1, [0, 1, 4])
    
    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-3"], 6, [0, 6, 7])

    with pytest.raises(ValueError, match=pattern):
        non_spreading_dirlaynet_minbudget(graphs["Dirlay_Graph-4"], 3, [1, 3, 5, 7])

    

# Test 1
# graph building
graph_1 = graphs["Dirlay_Graph-1"]
layers_1 = adjust_nodes_capacity(graph_1, 0) # src is 0 
targets_1 = [1, 2, 3]  # saving 1,2,3
G1 = create_st_graph(graph_1, targets_1)
reduction_G1 = graph_flow_reduction(G1, 0)
N_1_groups = min_cut_N_groups(reduction_G1, 0,layers_1)
matrix_1 = calculate_vaccine_matrix(layers_1, N_1_groups)
integer_matrix_1 = matrix_to_integers_values(matrix_1) 
min_budget_1 = min_budget_calculation(integer_matrix_1)

# Test 2
# graph building
graph_2 = graphs["Dirlay_Graph-2"]
layers_2 = adjust_nodes_capacity(graph_2, 0)  # src is 0
targets_2 = [2, 4]  # saving 2,4
G2 = create_st_graph(graph_2, targets_2)
reduction_G2 = graph_flow_reduction(G2, 0) 
N_2_groups = min_cut_N_groups(reduction_G2, 0, layers_2)
matrix_2 = calculate_vaccine_matrix(layers_2, N_2_groups)
integer_matrix_2 = matrix_to_integers_values(matrix_2)
min_budget_2 = min_budget_calculation(integer_matrix_2)

# Test 3
# graph building
graph_3 = graphs["Dirlay_Graph-3"]
layers_3 = adjust_nodes_capacity(graph_3, 0)  # src is 0
targets_3 = [1, 5, 7]  # saving 1,5,7
G3 = create_st_graph(graph_3, targets_3)
reduction_G3 = graph_flow_reduction(G3, 0)
N_3_groups = min_cut_N_groups(reduction_G3, 0, layers_3)
matrix_3 = calculate_vaccine_matrix(layers_3, N_3_groups)
integer_matrix_3 = matrix_to_integers_values(matrix_3)
min_budget_3 = min_budget_calculation(integer_matrix_3)

# Test 4
# graph building
graph_4 = graphs["Dirlay_Graph-4"]
layers_4 = adjust_nodes_capacity(graph_4, 0)  # src is 0
targets_4 = [4, 5, 6, 8]  # saving 4,5,6,8
G4 = create_st_graph(graph_4, targets_4)
reduction_G4 = graph_flow_reduction(G4, 0)
N_4_groups = min_cut_N_groups(reduction_G4, 0, layers_4)
matrix_4 = calculate_vaccine_matrix(layers_4, N_4_groups)
integer_matrix_4 = matrix_to_integers_values(matrix_4)
min_budget_4 = min_budget_calculation(integer_matrix_4)

def test_adjust_nodes_capacity():
    """
    This test checks if the node capacity and layers are correct.
    """
    # Tolerance for floating point comparisons, this is simply to not inturrpt in the after point shenanings.
    tolerance = 1e-6

    # Test 1
    layers_1_check = [[0], [1, 2], [3], [4], [5]]
    assert adjust_nodes_capacity(graph_1, 0) == layers_1_check
    assert math.isclose(graph_1.nodes[1]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3 + 1/4)), rel_tol=tolerance)
    assert math.isclose(graph_1.nodes[2]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3 + 1/4)), rel_tol=tolerance)
    assert math.isclose(graph_1.nodes[3]['capacity'], 1 / (2 * (1/1 + 1/2 + 1/3 + 1/4)), rel_tol=tolerance)
    assert math.isclose(graph_1.nodes[4]['capacity'], 1 / (3 * (1/1 + 1/2 + 1/3 + 1/4)), rel_tol=tolerance)
    assert math.isclose(graph_1.nodes[5]['capacity'], 1 / (4 * (1/1 + 1/2 + 1/3 + 1/4)), rel_tol=tolerance)

    # Test 2
    layers_2_check = [[0], [1, 2], [4, 3]]
    assert adjust_nodes_capacity(graph_2, 0) == layers_2_check
    assert math.isclose(graph_2.nodes[1]['capacity'], 1 / (1 * (1/1 + 1/2)), rel_tol=tolerance)
    assert math.isclose(graph_2.nodes[2]['capacity'], 1 / (1 * (1/1 + 1/2)), rel_tol=tolerance)
    assert math.isclose(graph_2.nodes[3]['capacity'], 1 / (2 * (1/1 + 1/2)), rel_tol=tolerance)
    assert math.isclose(graph_2.nodes[4]['capacity'], 1 / (2 * (1/1 + 1/2)), rel_tol=tolerance)

    # Test 3
    layers_3_check = [[0], [1, 2, 3], [5, 4], [6, 7]]
    assert adjust_nodes_capacity(graph_3, 0) == layers_3_check
    assert math.isclose(graph_3.nodes[1]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)
    assert math.isclose(graph_3.nodes[2]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)
    assert math.isclose(graph_3.nodes[3]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)
    assert math.isclose(graph_3.nodes[4]['capacity'], 1 / (2 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)
    assert math.isclose(graph_3.nodes[5]['capacity'], 1 / (2 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)
    assert math.isclose(graph_3.nodes[6]['capacity'], 1 / (3 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)
    assert math.isclose(graph_3.nodes[7]['capacity'], 1 / (3 * (1/1 + 1/2 + 1/3)), rel_tol=tolerance)

    # Test 4
    layers_4_check = [[0], [1, 2], [3, 4], [5], [6], [8]]
    assert adjust_nodes_capacity(graph_4, 0) == layers_4_check
    assert math.isclose(graph_4.nodes[1]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)
    assert math.isclose(graph_4.nodes[2]['capacity'], 1 / (1 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)
    assert math.isclose(graph_4.nodes[3]['capacity'], 1 / (2 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)
    assert math.isclose(graph_4.nodes[4]['capacity'], 1 / (2 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)
    assert math.isclose(graph_4.nodes[5]['capacity'], 1 / (3 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)
    assert math.isclose(graph_4.nodes[6]['capacity'], 1 / (4 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)
    assert math.isclose(graph_4.nodes[8]['capacity'], 1 / (5 * (1/1 + 1/2 + 1/3 + 1/4 + 1/5)), rel_tol=tolerance)



def test_create_st_graph() : 
    """
    Creates the s-t graph and connects the nodes we want to save.
    """
    # Test1
    # edges check
    assert "t" in G1
    assert G1.has_edge(1, "t")
    assert G1.has_edge(2, "t")
    assert G1.has_edge(3, "t")

    # Test2
    # edges check
    assert "t" in G2
    assert G2.has_edge(2, "t")
    assert G2.has_edge(4, "t")

    # Test3
    # edges check
    assert "t" in G3
    assert G3.has_edge(1, "t")
    assert G3.has_edge(5, "t")
    assert G3.has_edge(7, "t")

    # Test4
    # edges check
    assert "t" in G4
    assert G4.has_edge(4, "t")
    assert G4.has_edge(5, "t")
    assert G4.has_edge(6, "t")
    assert G4.has_edge(8, "t")

def test_min_cut_N_groups(): 
    """
    This test validates the nodes taken from the min-cut create the right groups (N_1...N_l).
    """
    def sort_dict_values(d): #| this is for sorting purposes , cases where [1,2] or [2,1] it does not really matters as we need to vaccinate them.
        return {k: sorted(v) for k, v in d.items()}

    # Test 1 
    # checking equality
    N1_groups_check = {1: [1, 2], 2: [], 3: [], 4: []}
    result_1 = min_cut_N_groups(reduction_G1, 0, layers_1)
    assert sort_dict_values(result_1) == sort_dict_values(N1_groups_check)

    # Test 2
    # checking equality
    N2_groups_check = {1: [2], 2: [4]}
    result_2 = min_cut_N_groups(reduction_G2, 0, layers_2)
    assert sort_dict_values(result_2) == sort_dict_values(N2_groups_check)

    # Test 3
    # checking equality
    N3_groups_check = {1: [1], 2: [], 3: [7]}
    result_3 = min_cut_N_groups(reduction_G3, 0, layers_3)
    assert sort_dict_values(result_3) == sort_dict_values(N3_groups_check)

    # Test 4
    # checking equality
    N4_groups_check = {1: [], 2: [4], 3: [5], 4: [], 5: []}
    result_4 = min_cut_N_groups(reduction_G4, 0, layers_4)
    assert sort_dict_values(result_4) == sort_dict_values(N4_groups_check)
    
def test_calculate_vaccine_matrix():
    """
    This test checks that the calculations made to create the triangular matrix from the min-cut nodes are correct.
    A matrix is valid if, for any column j, the column sum is exactly |Nj|.
    """
    # Test 1
    # checking equality
    matrix_1_check = np.array([[2, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
    np.testing.assert_array_equal(calculate_vaccine_matrix(layers_1, N_1_groups), matrix_1_check)

    # Test 2
    
    """
    Applying the formula: Mij ': =  Nj j , 1≤i≤j≤2 (l=2)
    N2_groups_check = {1: [2], 2: [4]} 
    N1 : = {2} |N1| = 1
    N2 : = {4} |N2| = 1

    1. M11 : i = 1, j = 1
    |N1|/1 = 1/1 = 1
    2. M12 : i = 1, j = 2
    |N2|/2 = 1/2
    3. M22 : i = 2, j = 2
    |N2|/2 = 1/2 
    """

    # checking equality
    matrix_2_check = np.array([[1, 0.5],
                               [0, 0.5]])
    np.testing.assert_array_equal(calculate_vaccine_matrix(layers_2, N_2_groups), matrix_2_check)

    # Test 3
    """
    Applying the formula: Mij ': =  Nj j , 1≤i≤j≤2 (l=2)
    N3_groups_check = {1: [1], 2: [], 3: [7]}
    N1 : = {1} |N1| = 1
    N2 : = {0} |N2| = 0
    N3 := {7} |N3| = 1

    1. M11 : i = 1, j = 1
    |N1|/1 = 1/1 = 1
    2. M12 : i = 1, j = 2
    |N2|/2 = 0/2 = 0
    3. M13 : i = 1, j = 3
    |N3|/3 = 1/3 
    4. M22 : i = 2, j = 2
    |N2|/2 = 0/2 = 0
    5. M23 : i = 2 j = 3
    |N3|/3 = 1/3 
    5. M33 : i = 2 j = 3
    |N3|/3 = 1/3 
    """
    # checking equality
    matrix_3_check = np.array([[1, 0, 1/3],
                               [0, 0, 1/3],
                               [0, 0, 1/3]])
    np.testing.assert_array_equal(calculate_vaccine_matrix(layers_3, N_3_groups), matrix_3_check)

    # Test 4
    # checking equality
    matrix_4_check = np.array([[0, 0.5, 1/3, 0, 0],
                               [0, 0.5, 1/3, 0, 0],
                               [0, 0, 1/3, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(calculate_vaccine_matrix(layers_4, N_4_groups), matrix_4_check)
    
def test_matrix_to_integers_values(): 
    """
    Test the matrix to integers values function.
    """
    # Test 1
    # checking equality
    matrix_1_check = np.array([[2, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
    
    expected_matrix_1 = np.array([[2, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    np.testing.assert_array_equal(matrix_to_integers_values(matrix_1_check), expected_matrix_1)

    # Test 2
    # checking equality
    matrix_2_check = np.array([[1, 0.5],
                               [0, 0.5]])
    
    expected_matrix_2 = np.array([[1, 1],
                               [0, 0]])
    np.testing.assert_array_equal(matrix_to_integers_values(matrix_2_check), expected_matrix_2)

    # Test 3
    # checking equality
    matrix_3_check = np.array([[1, 0, 1/3],
                               [0, 0, 1/3],
                               [0, 0, 1/3]])
    
    expected_matrix_3 = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [0, 0, 0]])
    np.testing.assert_array_equal(matrix_to_integers_values(matrix_3_check), expected_matrix_3)


def test_min_budget_calculation():
    """
    This test validates that the minimum budget is accurate.
    """
    # Test 1
    # checking equality
    matrix_1_check = np.array([[2, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
    assert min_budget_calculation(matrix_1_check) == 2

    # Test 2
    # checking equality
    matrix_2_check = np.array([[1, 1], #on time i = 1, vaccinate M11(1) nodes from layer 1. , on time i = 2, vaccinante M12(1) nodes from layer 2
                               [0, 0]])
    assert min_budget_calculation(matrix_2_check) == 1

    # Test 3
    # checking equality
    matrix_3_check = np.array([[1, 0, 1],
                               [0, 0, 0],
                               [0, 0, 0]])
    assert min_budget_calculation(matrix_3_check) == 1