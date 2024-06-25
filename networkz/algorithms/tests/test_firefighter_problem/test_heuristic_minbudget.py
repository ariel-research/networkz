import pytest
import networkx as nx
import json
import random
import time

from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import spreading_minbudget, heuristic_minbudget
from networkz.algorithms.approximation.firefighter_problem.Utils import parse_json_to_networkx

# Open the log file
log_file_path = 'networkz/algorithms/tests/test_firefighter_problem/comparisons/test_heuristic_minbudget_comparison.txt'

def log_message(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

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
    with open("networkz/algorithms/tests/test_firefighter_problem/graphs.json", "r") as file:
        json_data = json.load(file)
    graphs = parse_json_to_networkx(json_data)
    return graphs

graphs = get_graphs()

def test_source_not_in_graph():
        
        with pytest.raises(ValueError, match="Error: The source node isn't on the graph"):
            heuristic_minbudget(graphs["RegularGraph_Graph-1"], -3, [1, 0, 4, 5, 2, 6], True)

        with pytest.raises(ValueError, match="Error: The source node isn't on the graph"):
            heuristic_minbudget(graphs["RegularGraph_Graph-4"], 10, [1, 3, 5, 6, 7], False)

        with pytest.raises(ValueError, match="Error: The source node isn't on the graph"):
            heuristic_minbudget(graphs["RegularGraph_Graph-6"], 12, [9, 2, 3, 4, 6, 7], True)

        with pytest.raises(ValueError, match="Error: The source node isn't on the graph"):
            heuristic_minbudget(graphs["RegularGraph_Graph-8"], -1, [7, 10, 4, 9, 3, 11, 2], False)

        with pytest.raises(ValueError, match="Error: The source node isn't on the graph"):
            heuristic_minbudget(graphs["RegularGraph_Graph-3"], 8, [1, 4, 2], True)

def test_target_not_in_graph():
        
        with pytest.raises(ValueError, match="Error: Not all nodes in the targets list are on the graph."):
            heuristic_minbudget(graphs["RegularGraph_Graph-2"], 2, [0, 4, 5, 11, 6], True)

        with pytest.raises(ValueError, match="Error: Not all nodes in the targets list are on the graph."):
            heuristic_minbudget(graphs["RegularGraph_Graph-3"], 3, [0, 4, 5, -1, 1, 2], False)

        with pytest.raises(ValueError, match="Error: Not all nodes in the targets list are on the graph."):
            heuristic_minbudget(graphs["RegularGraph_Graph-6"], 7, [9, 2, 4, 5, 8, 11], True)

        with pytest.raises(ValueError, match="Error: Not all nodes in the targets list are on the graph."):
            heuristic_minbudget(graphs["RegularGraph_Graph-8"], 10, [0, 2, 4, 5, 8, 11, 12, 3, 15], False)

        with pytest.raises(ValueError, match="Error: Not all nodes in the targets list are on the graph."):
            heuristic_minbudget(graphs["RegularGraph_Graph-7"], 1, [3, 5, 4, 0, 13], True)

def test_source_is_target():
        
        with pytest.raises(ValueError, match="Error: The source node can't be a part of the targets list, since the virus is spreading from the source"):
            heuristic_minbudget(graphs["RegularGraph_Graph-1"], 0, [1, 2, 3, 0, 4, 5, 6], True)

        with pytest.raises(ValueError, match="Error: The source node can't be a part of the targets list, since the virus is spreading from the source"):
            heuristic_minbudget(graphs["RegularGraph_Graph-3"], 1, [5, 1, 4], False)

        with pytest.raises(ValueError, match="Error: The source node can't be a part of the targets list, since the virus is spreading from the source"):
            heuristic_minbudget(graphs["RegularGraph_Graph-4"], 4, [1, 2, 3, 4, 5, 6, 7], True)

        with pytest.raises(ValueError, match="Error: The source node can't be a part of the targets list, since the virus is spreading from the source"):
            heuristic_minbudget(graphs["RegularGraph_Graph-6"], 0, [0, 3, 5, 6, 7, 8, 9], False)

        with pytest.raises(ValueError, match="Error: The source node can't be a part of the targets list, since the virus is spreading from the source"):
            heuristic_minbudget(graphs["RegularGraph_Graph-8"], 0, [13, 10, 8, 6, 5, 4, 3, 0, 1, 2], True)


def test_save_all_vertices_spreading():
    log_message("Starting test_save_all_vertices_spreading:\n")
    try:
        for graph_name in graphs:
            G = graphs[graph_name]
            source = 0
            targets = [n for n in G.nodes if n != source]

            start_time = time.time()
            spreading_result = spreading_minbudget(G, source, targets)[0]
            spreading_time = time.time() - start_time

            start_time = time.time()
            heuristic_result = heuristic_minbudget(G, source, targets, True)[0]
            heuristic_time = time.time() - start_time

            log_message(f"{graph_name} - Spreading Result: {spreading_result}, Heuristic Result: {heuristic_result}")
            log_message(f"{graph_name} - Spreading Time: {spreading_time:.6f}s, Heuristic Time: {heuristic_time:.6f}s")

            assert spreading_result >= heuristic_result

    finally:
        log_message("\nFinished test_save_all_vertices_spreading.\n---------------------------------------------------------------------------------------------------------------------------")

def test_save_subgroup_vertices_spreading():
    log_message("Starting test_save_subgroup_vertices_spreading:\n")
    try:
        for graph_name in graphs:
            G = graphs[graph_name]
            source = 0
            targets = [n for n in G.nodes if random.random() < 0.5 and n != source]

            start_time = time.time()
            spreading_result = spreading_minbudget(G, source, targets)[0]
            spreading_time = time.time() - start_time

            start_time = time.time()
            heuristic_result = heuristic_minbudget(G, source, targets, True)[0]
            heuristic_time = time.time() - start_time

            log_message(f"{graph_name} (subgroup) - Spreading Result: {spreading_result}, Heuristic Result: {heuristic_result}")
            log_message(f"{graph_name} (subgroup) - Spreading Time: {spreading_time:.6f}s, Heuristic Time: {heuristic_time:.6f}s")

            assert spreading_result >= heuristic_result

           
    finally:
        log_message("\nFinished test_save_subgroup_vertices_spreading.\n---------------------------------------------------------------------------------------------------------------------------")

def test_random_graph_comparison():
    log_message("Starting test_random_graph_comparison:\n")
    try:
        for i in range(10):
            num_nodes = random.randint(2, 100)
            nodes = list(range(num_nodes + 1))
            num_edges = 1000
            save_amount = random.randint(1, num_nodes)
            targets = []
            G = nx.DiGraph()

            G.add_nodes_from(nodes, status="target")
            for _ in range(num_edges):
                source = random.randint(0, num_nodes - 1)
                target = random.randint(0, num_nodes - 1)
                if source != target:  # Ensure no self-loops
                    G.add_edge(source, target)
            for node in range(save_amount):
                probability = random.random()
                if probability < 0.75 and node != 0:
                    targets.append(node)

            start_time = time.time()
            spreading_answer = spreading_minbudget(G, 0, targets)[0]
            spreading_time = time.time() - start_time

            start_time = time.time()
            heuristic_answer = heuristic_minbudget(G, 0, targets, True)[0]
            heuristic_time = time.time() - start_time

            
            log_message(f"Random Graph {i+1} - Spreading Result: {spreading_answer}, Heuristic Result: {heuristic_answer}")
            log_message(f"Random Graph {i+1} - Spreading Time: {spreading_time:.6f}s, Heuristic Time: {heuristic_time:.6f}s")

            assert heuristic_answer <= spreading_answer

    finally:
        log_message("\nFinished test_random_graph_comparison.\n---------------------------------------------------------------------------------------------------------------------------")
