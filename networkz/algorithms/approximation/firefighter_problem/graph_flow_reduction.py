# graph_flow_transformations.py
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def max_flow_with_node_capacity(graph: nx.DiGraph, source: int = None, target: int = None) -> nx.DiGraph:
    """
    Computes a maximum flow in the given graph, where each node has a capacity

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
    nx.DiGraph
        The transformed graph after flow reduction.

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
    >>> G.add_node(1, capacity=10)
    >>> G.add_node(2, capacity=15)
    >>> G.add_node(3, capacity=20)
    >>> G.add_edge(1, 2)
    >>> G.add_edge(2, 3)
    >>> G.add_edge(1, 3)
    >>> H = max_flow_with_node_capacity(G, 1, 3)
    >>> sorted(list(H.nodes))
    ['1_in', '1_out', '2_in', '2_out', '3_in', '3_out']
    >>> sorted(list(H.edges(data=True)))
    [('1_in', '1_out', {'weight': inf}), ('1_out', '2_in', {'weight': inf}), ('1_out', '3_in', {'weight': inf}), ('2_in', '2_out', {'weight': 15}), ('2_out', '3_in', {'weight': inf}), ('3_in', '3_out', {'weight': inf})]
    >>> H = max_flow_with_node_capacity(G)
    >>> sorted(list(H.nodes))
    ['1_in', '1_out', '2_in', '2_out', '3_in', '3_out']
    >>> sorted(list(H.edges(data=True)))
    [('1_in', '1_out', {'weight': 10}), ('1_out', '2_in', {'weight': inf}), ('1_out', '3_in', {'weight': inf}), ('2_in', '2_out', {'weight': 15}), ('2_out', '3_in', {'weight': inf}), ('3_in', '3_out', {'weight': 20})]
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
    return H

if __name__ == "__main__":
    import doctest
    result = doctest.testmod(verbose=False)
    print(f"Doctest results: {result}")
