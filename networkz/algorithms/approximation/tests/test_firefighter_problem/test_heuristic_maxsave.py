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
import logging
import os
import time
from datetime import datetime

from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import heuristic_maxsave, spreading_maxsave
from networkz.algorithms.approximation.firefighter_problem.Utils import find_best_neighbor, parse_json_to_networkx, Status

def setup_logger():
    logger = logging.getLogger('firefighter_problem_tests')
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

path_to_graphs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs.json')
if os.path.exists(path_to_graphs):
    with open(path_to_graphs, "r") as file:
        json_data = json.load(file)
else:
    raise FileNotFoundError(f"{path_to_graphs} does not exist.")
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
    logger.info(f"Testing strategy length for {graph_key}")
    graph = graphs[graph_key]
    calculated_strategy = spreading_maxsave(graph, budget, source, targets)[0]
    logger.info(f"Calculated strategy: {calculated_strategy}")
    
    assert len(calculated_strategy) == expected_length
    logger.info(f"Strategy length test passed for {graph_key}")

@pytest.mark.parametrize("graph_key, budget, source, targets, expected_strategy", [
    ("RegularGraph_Graph-1", 1, 0, [1, 2, 3, 4, 5, 6], [(1, 1), (6, 2)]),
    ("Dirlay_Graph-5", 2, 0, [1, 2, 3, 4, 5, 6, 7, 8], [(5, 1), (2, 1)]),
])
def test_save_all_vertices(graph_key, budget, source, targets, expected_strategy):
    logger.info(f"Testing save all vertices for {graph_key}")
    graph = graphs[graph_key]
    calculated_strategy = heuristic_maxsave(graph, budget, source, targets)[0]
    logger.info(f"Calculated strategy: {calculated_strategy}")
    
    assert calculated_strategy == expected_strategy
    logger.info(f"Save all vertices test passed for {graph_key}")

@pytest.mark.parametrize("graph_key, budget, source, targets, expected_strategy", [
    ("RegularGraph_Graph-6", 2, 1, [3, 9, 0, 5, 6], [(2, 1)]),
    ("RegularGraph_Graph-4", 1, 0, [2, 6, 4], [(1, 1)]),
])
def test_save_subgroup_vertices(graph_key, budget, source, targets, expected_strategy):
    logger.info(f"Testing save subgroup vertices for {graph_key}")
    graph = graphs[graph_key]
    calculated_strategy = heuristic_maxsave(graph, budget, source, targets)[0]
    logger.info(f"Calculated strategy: {calculated_strategy}")
    
    assert calculated_strategy == expected_strategy
    logger.info(f"Save subgroup vertices test passed for {graph_key}")

def test_random_graph_comparison():
    logger.info("Starting test_random_graph_comparison:")
    try:
        for i in range(10):
            num_nodes = random.randint(2, 100)
            nodes = list(range(num_nodes + 1))
            num_edges = 1000
            save_amount = random.randint(1, num_nodes)
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
                if probability < 0.75 and node != 0:
                    targets.append(node)
            
            logger.info(f"Random Graph {i+1} - Targets: {targets}")
            
            start_time = time.time()
            spreading_answer = spreading_maxsave(G, 1, 0, targets)[1]
            spreading_time = time.time() - start_time
            
            start_time = time.time()
            heuristic_answer = heuristic_maxsave(G, 1, 0, targets)[1]
            heuristic_time = time.time() - start_time
            
            logger.info(f"Random Graph {i+1} - Spreading Result: {len(spreading_answer)}, Heuristic Result: {len(heuristic_answer)}")
            logger.info(f"Random Graph {i+1} - Spreading Time: {spreading_time:.6f}s, Heuristic Time: {heuristic_time:.6f}s")
            
            if len(spreading_answer) > len(heuristic_answer):
                warning_message = f"Warning: Heuristic result ({len(heuristic_answer)}) is less than spreading result ({len(spreading_answer)}) for Random Graph {i+1}"
                logger.warning(warning_message)

    finally:
        logger.info("Finished test_random_graph_comparison.")
        logger.info("-" * 100)


if __name__ == "__main__":
     pytest.main(["-v",__file__])

