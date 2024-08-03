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

from networkz.algorithms.approximation.firefighter_problem.Utils import parse_json_to_networkx, Status

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

@pytest.fixture
def missing_vertices_json():
    return {
        "InvalidGraph": {
            "Graph-1": {
                "edges": [[0, 1], [1, 2]]
            }
        }
    }

@pytest.fixture
def missing_edges_json():
    return {
        "InvalidGraph": {
            "Graph-2": {
                "vertices": [0, 1, 2]
            }
        }
    }

@pytest.fixture
def empty_json():
    return {
        "InvalidGraph": {
            "Graph-3": {
                "vertices": [],
                "edges": []
            }
        }
    }


def test_parsing_dirlay_graph(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    dirlay_graph = graphs["Dirlay_Graph-1"]
    assert isinstance(dirlay_graph, nx.DiGraph)

def test_parsing_dirlay_graph_nodes(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    dirlay_graph = graphs["Dirlay_Graph-1"]
    assert set(dirlay_graph.nodes()) == {0, 1, 2, 3, 4, 5}

def test_parsing_dirlay_graph_edges(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    dirlay_graph = graphs["Dirlay_Graph-1"]
    assert set(dirlay_graph.edges()) == {(0, 1), (0, 2)}

def test_parsing_regular_graph(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    regular_graph = graphs["RegularGraph_Graph-1"]
    assert isinstance(regular_graph, nx.Graph)

def test_parsing_regular_graph_nodes(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    regular_graph = graphs["RegularGraph_Graph-1"]
    assert set(regular_graph.nodes()) == {0, 1, 2}

def test_parsing_regular_graph_edges(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    regular_graph = graphs["RegularGraph_Graph-1"]
    assert set(regular_graph.edges()) == {(0, 1), (1, 2)}
    
def test_parse_exceptions_missing_vertices(missing_vertices_json):
    with pytest.raises(KeyError):
        parse_json_to_networkx(missing_vertices_json)

def test_parse_exceptions_missing_edges(missing_edges_json):
    with pytest.raises(KeyError):
        parse_json_to_networkx(missing_edges_json)

def test_parse_exceptions_empty_json(empty_json):
    with pytest.raises(KeyError):
        parse_json_to_networkx(empty_json)
        
def test_parsing_dirlay_graph_status(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    dirlay_graph = graphs["Dirlay_Graph-1"]
    for node in dirlay_graph.nodes(data=True):
        assert node[1]["status"] == Status.VULNERABLE.value

def test_parsing_regular_graph_status(sample_json_data):
    graphs = parse_json_to_networkx(sample_json_data)

    regular_graph = graphs["RegularGraph_Graph-1"]
    for node in regular_graph.nodes(data=True):
        assert node[1]["status"] == Status.VULNERABLE.value

if __name__ == "__main__":
    pytest.main(["-v", __file__])