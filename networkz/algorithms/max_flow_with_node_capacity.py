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
import networkx.algorithms.connectivity as algo 
from networkz.algorithms.approximation.firefighter_problem.Utils import *
import logging


logger = logging.getLogger(__name__)

def min_cut_with_node_capacity(graph: nx.DiGraph, source: int = None, target: int = None) -> set:
    """
    Computes a minimum cut in the given graph, where each node has a capacity

    Parameters:
    ----------
    graph : nx.DiGraph
        The original directed graph. Each node should have a 'capacity' attribute.
    source (Optional) : int
        The source node in the graph. This node will have an infinite capacity 
        between its "in" and "out" nodes.
    target (Optional) : int
        The target node in the graph. This node will have an infinite capacity 
        between its "in" and "out" nodes

    Returns:
    -------
    set
       The minimum cut nodes 

    Notes:
    -----
    This function transforms a given directed graph into a new graph where each node
    is split into two nodes (an "in" node and an "out" node) connected by an edge 
    with a capacity (weight). The transformation is used for flow problems where 
    nodes have capacities instead of edges - allowing to run algorithms which
    were originally designed to be used for edge-flow problems.
    - If a node does not have a 'capacity' attribute, a default capacity of 1 
      is used.
    - There is infinite capacity between two different edge_out & edge_in

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_node(1, capacity=0.6)
    >>> G.add_node(2, capacity=0.6)
    >>> G.add_node(3, capacity=0.3)
    >>> G.add_node(4, capacity=0.3)
    >>> G.add_edge(0, 1)
    >>> G.add_edge(0, 2)
    >>> G.add_edge(2, 3)
    >>> G.add_edge(1, 4)
    >>> s_t_G = create_st_graph(G, [2,4], 't')
    >>> min_cut_nodes = min_cut_with_node_capacity(s_t_G, 0, 4)
    >>> sorted(min_cut_nodes)
    ['2_out', '4_out']
    """
    logger.info("Starting graph flow reduction")
    H = nx.DiGraph()
    
    logger.debug("Adding nodes and internal edges")
    for node in graph.nodes:
        in_node, out_node = f'{node}_in', f'{node}_out'
        H.add_nodes_from([in_node, out_node])
        if node == source:
            H.add_edge(in_node, out_node, weight=float('inf'))
            logger.debug(f"Added infinite capacity edge for source node: {node}")
        elif node == target:
            H.add_edge(in_node, out_node, weight=float('inf'))
            logger.debug(f"Added infinite capacity edge for target node: {node}")
        else:
            capacity = graph.nodes[node].get('capacity', 1)
            H.add_edge(in_node, out_node, weight=capacity)
            logger.debug(f"Added edge with capacity {capacity} for node: {node}")
    
    logger.debug("Adding edges between nodes")
    for edge in graph.edges:
        u_out, v_in = f'{edge[0]}_out', f'{edge[1]}_in'
        H.add_edge(u_out, v_in, weight=float('inf'))
        logger.debug(f"Added infinite capacity edge from {u_out} to {v_in}")
    
    logger.info("Graph flow reduction finished")

    # Compute the minimum cut
    logger.info(f"Finding the minimum cut on the graph after reduction") 
    min_cut_nodes = algo.minimum_st_node_cut(H, f'{source}_out', 't_in')
    
    logger.info(f"Minimum Cut is: {min_cut_nodes}")  
    return min_cut_nodes

if __name__ == "__main__":
    import doctest
    result = doctest.testmod(verbose=False)
    print(f"Doctest results: {result}")
