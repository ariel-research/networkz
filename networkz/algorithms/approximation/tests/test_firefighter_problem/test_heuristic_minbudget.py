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
import random
import time
import logging
import os
import json

from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import spreading_minbudget, heuristic_minbudget
from networkz.algorithms.approximation.firefighter_problem.Random_Graph_Generator import generate_random_DiGraph
from networkz.algorithms.approximation.firefighter_problem.Utils import parse_json_to_networkx


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


@pytest.fixture
def sample_json_data():
    return {
        "Dirlay": {
            "Graph-1": {
                "vertices": [0, 1, 2, 3, 4, 5],
                "edges": [[0, 1], [0, 2]]
            },
        },
        "RegularGraph": {
            "Graph-1": {
                "vertices": [0, 1, 2],
                "edges": [[0, 1], [1, 2]]
            },
        }
    }

def get_graphs():
    path_to_graphs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs.json')
    if os.path.exists(path_to_graphs):
        with open(path_to_graphs, "r") as file:
            json_data = json.load(file)
    else:
        raise FileNotFoundError(f"{path_to_graphs} does not exist.")
    
    graphs = parse_json_to_networkx(json_data)
    return graphs

graphs = get_graphs()

@pytest.mark.parametrize("G, source, targets, directed", [
    (graphs["RegularGraph_Graph-1"], -3, [1, 0, 4, 5, 2, 6], True),
    (graphs["RegularGraph_Graph-4"], 10, [1, 3, 5, 6, 7], False),
    (graphs["RegularGraph_Graph-6"], 12, [9, 2, 3, 4, 6, 7], True),
    (graphs["RegularGraph_Graph-8"], -1, [7, 10, 4, 9, 3, 11, 2], False),
    (graphs["RegularGraph_Graph-3"], 8, [1, 4, 2], True)
])
def test_source_not_in_graph(G, source, targets, directed):
    with pytest.raises(ValueError, match="Error: The source node isn't on the graph"):
        heuristic_minbudget(G, source, targets, directed)

@pytest.mark.parametrize("G, source, targets, directed", [
    (graphs["RegularGraph_Graph-2"], 2, [0, 4, 5, 11, 6], True),
    (graphs["RegularGraph_Graph-3"], 3, [0, 4, 5, -1, 1, 2], False),
    (graphs["RegularGraph_Graph-6"], 7, [9, 2, 4, 5, 8, 11], True),
    (graphs["RegularGraph_Graph-8"], 10, [0, 2, 4, 5, 8, 11, 12, 3, 15], False),
    (graphs["RegularGraph_Graph-7"], 1, [3, 5, 4, 0, 13], True)
])
def test_target_not_in_graph(G, source, targets, directed):
    with pytest.raises(ValueError, match="Error: Not all nodes in the targets list are on the graph."):
        heuristic_minbudget(G, source, targets, directed)

@pytest.mark.parametrize("G, source, targets, directed", [
    (graphs["RegularGraph_Graph-1"], 0, [1, 2, 3, 0, 4, 5, 6], True),
    (graphs["RegularGraph_Graph-3"], 1, [5, 1, 4], False),
    (graphs["RegularGraph_Graph-4"], 4, [1, 2, 3, 4, 5, 6, 7], True),
    (graphs["RegularGraph_Graph-6"], 0, [0, 3, 5, 6, 7, 8, 9], False),
    (graphs["RegularGraph_Graph-8"], 0, [13, 10, 8, 6, 5, 4, 3, 0, 1, 2], True)
])
def test_source_is_target(G, source, targets, directed):
    with pytest.raises(ValueError, match="Error: The source node can't be a part of the targets list, since the virus is spreading from the source"):
        heuristic_minbudget(G, source, targets, directed)


@pytest.mark.parametrize("i", range(10))
def test_random_graph_comparison(i):
    logger.info(f"Starting test_random_graph_comparison for Random Graph {i+1}:")
    try:
        G =  generate_random_DiGraph(num_nodes=100, edge_probability=0.5)

        num_targets = random.randint(1, int(len(G.nodes) / 2) + 1)
        nodes_to_sample = list(G.nodes)
        nodes_to_sample.remove(0)
        targets = random.sample(list(nodes_to_sample), num_targets)

        start_time = time.time()
        spreading_answer = spreading_minbudget(G, 0, targets)[0]
        spreading_time = time.time() - start_time

        start_time = time.time()
        heuristic_answer = heuristic_minbudget(G, 0, targets, True)[0]
        heuristic_time = time.time() - start_time

        logger.info(f"Random Graph {i+1} - Spreading Result: {spreading_answer}, Heuristic Result: {heuristic_answer}")
        logger.info(f"Random Graph {i+1} - Spreading Time: {spreading_time:.6f}s, Heuristic Time: {heuristic_time:.6f}s")

        if heuristic_answer > spreading_answer:
            warning_message = f"Warning: Heuristic result ({heuristic_answer}) is greater than spreading result ({spreading_answer}) for Random Graph {i+1}"
            logger.warning(warning_message)

    finally:
        logger.info(f"Finished test_random_graph_comparison for Random Graph {i+1}.")
        logger.info("-" * 100)

if __name__ == "__main__":
    pytest.main(["-v", __file__])