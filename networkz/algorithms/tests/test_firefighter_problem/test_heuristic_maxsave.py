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

import pytest
import networkx as nx
import json
import random

from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import heuristic_maxsave, spreading_maxsave
from networkz.algorithms.approximation.firefighter_problem.Utils import  find_best_neighbor, parse_json_to_networkx, Status

with open("networkz/algorithms/tests/test_firefighter_problem/graphs.json", "r") as file:
        json_data = json.load(file)
graphs = parse_json_to_networkx(json_data)

@pytest.mark.parametrize("graph_key, budget, source, targets", [
    ("RegularGraph_Graph-1", 1, -2, [1, 2, 3, 4, 5, 6]),
    ("RegularGraph_Graph-4", 1, 8, [1, 2, 4, 6, 7]),
    ("RegularGraph_Graph-6", 1, 10, [0, 2, 3, 5, 6, 7, 8, 9]),
    ("RegularGraph_Graph-8", 1, 17, [1, 7, 12, 14, 8, 3, 11, 2]),
    ("RegularGraph_Graph-3", 1, 6, [1, 3, 5]),
])
def test_source_not_in_graph(graph_key, budget, source, targets):
    with pytest.raises(ValueError):
        heuristic_maxsave(graphs[graph_key], budget, source, targets)

@pytest.mark.parametrize("graph_key, budget, source, targets", [
    ("RegularGraph_Graph-2", 1, 0, [1, 2, 3, 9, 5, 16]),
    ("RegularGraph_Graph-3", 1, 4, [1, 2, 3, 6, 7]),
    ("RegularGraph_Graph-6", 1, 3, [0, 2, 5, 6, 7, 8, 10]),
    ("RegularGraph_Graph-8", 1, 11, [1, 3, 12, 19, 8, 10, 4, 2]),
    ("RegularGraph_Graph-7", 1, 2, [1, 3, -1, 5]),
])
def test_target_not_in_graph(graph_key, budget, source, targets):
    with pytest.raises(ValueError):
        heuristic_maxsave(graphs[graph_key], budget, source, targets)

@pytest.mark.parametrize("graph_key, budget, source, targets", [
    ("RegularGraph_Graph-1", 1, 0, [1, 2, 3, 0, 4, 5, 6]),
    ("RegularGraph_Graph-3", 1, 1, [5, 1, 4]),
    ("RegularGraph_Graph-4", 1, 4, [1, 2, 3, 4, 5, 6, 7]),
    ("RegularGraph_Graph-6", 1, 0, [0, 3, 5, 6, 7, 8, 9]),
    ("RegularGraph_Graph-8", 1, 0, [13, 10, 8, 6, 5, 4, 3, 0, 1, 2]),
])
def test_source_is_target(graph_key, budget, source, targets):
    with pytest.raises(ValueError):
        heuristic_maxsave(graphs[graph_key], budget, source, targets)

@pytest.mark.parametrize("graph_key, budget, source, targets, expected_length", [
    ("RegularGraph_Graph-1", 1, 0, [1, 2, 3, 4, 5, 6], 2),
    ("Dirlay_Graph-5", 2, 0, [1, 2, 3, 4, 5, 6, 7, 8], 3),
])
def test_strategy_length(graph_key, budget, source, targets, expected_length):
    graph = graphs[graph_key]
    calculated_strategy = spreading_maxsave(graph, budget, source, targets)[0]
    print(calculated_strategy)
    
    assert len(calculated_strategy) == expected_length


@pytest.mark.parametrize("graph_key, budget, source, targets, expected_strategy", [
    ("RegularGraph_Graph-1", 1, 0, [1, 2, 3, 4, 5, 6], [(1, 1), (6, 2)]),
    ("Dirlay_Graph-5", 2, 0, [1, 2, 3, 4, 5, 6, 7, 8], [(5, 1), (2, 1)]),
])
def test_save_all_vertices(graph_key, budget, source, targets, expected_strategy):
    graph = graphs[graph_key]
    calculated_strategy = heuristic_maxsave(graph, budget, source, targets)[0]
    print(calculated_strategy)
    
    assert calculated_strategy == expected_strategy

@pytest.mark.parametrize("graph_key, budget, source, targets, expected_strategy", [
    ("RegularGraph_Graph-6", 2, 1, [3, 9, 0, 5, 6], [(2, 1)]),
    ("RegularGraph_Graph-4", 1, 0, [2, 6, 4], [(1, 1)]),
])
def test_save_subgroup_vertices(graph_key, budget, source, targets, expected_strategy):
    graph = graphs[graph_key]
    calculated_strategy = heuristic_maxsave(graph, budget, source, targets)[0]
    print(calculated_strategy)
    
    assert calculated_strategy == expected_strategy

def test_random_graph_comparison():
    for i in range(10):
        num_nodes = random.randint(2,100)
        nodes = list(range(num_nodes+1))
        num_edges = 1000
        save_amount = random.randint(1,num_nodes)
        targets = []
        G = nx.DiGraph()
        
        G.add_nodes_from(nodes, status=Status.VULNERABLE.value)
        for _ in range(num_edges):
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            if source != target:  # Ensure no self-loops
                G.add_edge(source, target)
        for node in range(save_amount):
            probability = random.random()
            if probability < 0.75 and node!=0:
                targets.append(node)
        
        print(targets)
        spreading_answer = spreading_maxsave(G,1,0,targets)[1]
        heuristic_answer = heuristic_maxsave(G,1,0,targets)[1]
        print(spreading_answer)
        print(heuristic_answer)
       
        assert len(spreading_answer) <= len(heuristic_answer)
    
    print("All tests have passed!")