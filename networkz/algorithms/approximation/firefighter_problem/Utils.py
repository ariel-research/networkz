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

import networkx as nx
import networkx.algorithms.connectivity as algo 
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import copy
import logging

class Status(Enum):
    VULNERABLE = "vulnerable"
    INFECTED = "infected"
    VACCINATED = "vaccinated"
    DIRECTLY_VACCINATED = "directly vaccinated"


node_colors = {
    'vulnerable': 'gray',
    'infected': 'red',
    'vaccinated': 'blue',
    'directly vaccinated': 'green',
    'default' : "#00FFD0"
}

logger = logging.getLogger(__name__)

def validate_parameters(graph:nx.DiGraph, source:int, targets:list)->None:
    """
    Validate the source and target nodes in the graph.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    source : int
        Source node.
    targets : list
        List of target nodes to save.

    Raises:
    -------
    ValueError
        If the source node is not in the graph.
        If the source node is in the targets list.
        If any target node is not in the graph.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([1, 2, 3, 4])
    >>> validate_parameters(G, 1, [2, 3])
    >>> validate_parameters(G, 5, [2, 3])
    Traceback (most recent call last):
        ...
    ValueError: Error: The source node isn't on the graph
    >>> validate_parameters(G, 1, [1, 3])
    Traceback (most recent call last):
        ...
    ValueError: Error: The source node can't be a part of the targets list, since the virus is spreading from the source
    >>> validate_parameters(G, 1, [2, 5])
    Traceback (most recent call last):
        ...
    ValueError: Error: Not all nodes in the targets list are on the graph.
    """
    graph_nodes = list(graph.nodes())
    if source not in graph_nodes:
        logger.critical("Error: The source node isn't on the graph")
        raise ValueError("Error: The source node isn't on the graph")
        
    if source in targets:
        logger.critical("Error: The source node can't be a part of the targets list, since the virus is spreading from the source")
        raise ValueError("Error: The source node can't be a part of the targets list, since the virus is spreading from the source")
        
    if not all(node in graph_nodes for node in targets):
        logger.critical("Error: Not all nodes in the targets list are on the graph.")
        raise ValueError("Error: Not all nodes in the targets list are on the graph.")

def is_st_layered_dag(graph: nx.DiGraph, s: any, t: any) -> bool: #TODO: make this work and incorporate into code
    """
    Validates if a given graph is an s-t directed layered network.

    In an s-t directed l-layered network, the vertex set consists of 
    V = (L0 := {s}) U L1 U ... U Ll U {t}, and all arcs except those 
    entering t are from a vertex in some layer Li to a vertex in Li+1; 
    arcs entering t may originate from any vertex other than t.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - s: Source vertex.
    - t: Target vertex.

    Returns:
    - True if the graph is an s-t directed l-layered network,
    - False otherwise.
    """
    logger.info(f'Validating {graph} is an s-t directed layered network')
    if not nx.is_directed_acyclic_graph(graph):
        logger.error(f'{graph} is not DAG and therefore cannot be an s-t directed layered network!')
        return False
    
    topo_sort = list(nx.topological_sort(graph))
    logger.debug(f'Performing topological sort to determine an ordering of vertices\n Sorted:{topo_sort}')
    
    if topo_sort[0] != s or topo_sort[-1] != t:
        logger.error(f'First node in topological sort is not "t" or last node is not "s"')
        return False

    # Assign layer indices based on topological sort
    node_layers = {node: index for index, node in enumerate(topo_sort)}
    
    # Check if all edges (except those entering t) go from a lower layer to the next higher layer
    for u, v in graph.edges():
        if v != t and node_layers[u] + 1 != node_layers[v]:
            logger.error(f'The edge {u},{v} is an edge that does not enter "t" but goes from a higher to a lower layer - violating the required property')
            return False

    return True

def is_dag(graph:nx.DiGraph) -> bool:
    """
    Validates if a given graph is a Directed Acyclic Graph (DAG) using NetworkX.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.

    Returns:
    -------
    bool
        True if the graph is a DAG, False otherwise.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3)])
    >>> is_dag(G)
    True
    >>> G.add_edge(3, 1)
    >>> is_dag(G)
    False
    """
    return nx.is_directed_acyclic_graph(graph)

# ============================ Spreading Max-Save ============================
def calculate_gamma(graph:nx.DiGraph, source:int, targets:list)-> dict:
    """
    Calculate Gamma and S(u,t) based on the calculation in the article.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    source : int
        Source node.
    targets : list
        List of target nodes to save.

    Returns:
    -------
    gamma (dict)
        Dictionary of vaccination options.
    direct_vaccination (dict)
        Dictionary of direct vaccination strategies - S(u,t).

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    >>> G.nodes[1]['capacity'] = 10
    >>> G.nodes[2]['capacity'] = 15
    >>> G.nodes[3]['capacity'] = 20
    >>> gamma, direct_vaccination = calculate_gamma(G, 1, [3])
    >>> gamma
    {2: [(2, 1)], 3: [(3, 1)]}
    >>> direct_vaccination
    {(2, 1): [], (3, 1): [3]}
    """
    gamma = {}
    direct_vaccination = {}
    unreachable_nodes = []
    path_length = dict(nx.all_pairs_shortest_path_length(graph))
    for key in graph.nodes:
        vaccination_options = []
        for node in graph.nodes:
                if path_length[source].get(key) is not None and path_length[node].get(key) is not None:
                    s_to_v = path_length[source].get(key)
                    u_to_v = path_length[node].get(key)
                    max_time = s_to_v - u_to_v
                    if max_time > 0:
                        for i in range(1,max_time+1):
                            option = (node,i)
                            vaccination_options.append(option)
                            if option not in direct_vaccination:
                                direct_vaccination[option] = []
                            if key in targets:
                                direct_vaccination[option].append(key)

        if not vaccination_options:
            unreachable_nodes.append(key)
        if key != source:                       
            gamma[key] = vaccination_options

    for strategy in direct_vaccination:
        for node in unreachable_nodes:
            if node in targets:
                direct_vaccination[strategy].append(node)
    
    logger.info("Gamma is: " + str(gamma))
    logger.info("S(u,t) is: " + str(direct_vaccination))
    return gamma, direct_vaccination

def calculate_epsilon(direct_vaccinations:dict)->list:
    """
    Calculate Epsilon based on the calculation in the article.

    Parameters:
    ----------
    direct_vaccinations : dict
        Dictionary of direct vaccination strategies.

    Returns:
    -------
    list
        List of direct vaccination groups by time step.

    Examples:
    --------
    >>> direct_vaccinations = {(1, 1): [3], (2, 2): [4], (3, 1): [5]}
    >>> calculate_epsilon(direct_vaccinations)
    [[(1, 1), (3, 1)], [(2, 2)]]
    """
    from itertools import groupby
    from operator import itemgetter

    sorted_dict = sorted(direct_vaccinations, key=itemgetter(1))
    epsilon = [list(group) for _, group in groupby(sorted_dict, key=itemgetter(1))]
    
    logger.info("Epsilon is: " + str(epsilon))
    return epsilon

def find_best_direct_vaccination(graph:nx.DiGraph, direct_vaccinations:dict, current_time_options:list, targets:list)->tuple:
    """
    Find the best direct vaccination strategy for the current time step that saves more new nodes in targets.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    direct_vaccinations : dict
        Dictionary of direct vaccination strategies.
    current_time_options : list
        List of current time step vaccination options.
    targets : list
        List of target nodes.

    Returns:
    -------
    tuple
        Best direct vaccination option and the nodes saved by this option.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([(1, {"status": 'vulnerable'}), (2, {"status": 'vulnerable'}), (3, {"status": 'vulnerable'})])
    >>> direct_vaccinations = {(1, 1): [1], (2, 2): [2]}
    >>> current_time_options = [(1, 1), (2, 2)]
    >>> targets = [2]
    >>> find_best_direct_vaccination(G, direct_vaccinations, current_time_options, targets)
    ((2, 2), {2})
    """
    best_vaccination = () 
    nodes_saved = []
    common_elements = None
    max_number = -1
    for option in current_time_options:
        if(graph.nodes[option[0]]['status'] == Status.VULNERABLE.value):
            nodes_list = direct_vaccinations.get(option)
            common_elements = set(nodes_list) & set(targets)
            logger.debug(f"Direct vaccination: {option}, Nodes saved: {common_elements} (if set(), then it's empty)")
            if len(common_elements) > max_number:
                best_vaccination = option
                nodes_saved = common_elements
                max_number = len(common_elements)
    
    if best_vaccination != ():
        logger.info("The best direct vaccination is: " + str(best_vaccination) + " and it saves nodes: " + str(nodes_saved))
    return best_vaccination, nodes_saved

def spread_virus(graph:nx.DiGraph, infected_nodes:list)->bool:
    """
    Spread the virus from infected nodes to their neighbors.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    infected_nodes : list
        List of currently infected nodes.

    Returns:
    -------
    bool
        True if there are new infections, False otherwise.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([(1, {"status": 'infected'}), (2, {"status": 'vulnerable'}), (3, {"status": 'vulnerable'})])
    >>> G.add_edges_from([(1, 2), (2, 3)])
    >>> infected_nodes = [1]
    >>> spread_virus(G, infected_nodes)
    True
    >>> G.nodes(data=True)
    NodeDataView({1: {'status': 'infected'}, 2: {'status': 'infected'}, 3: {'status': 'vulnerable'}})
    """
    new_infected_nodes = []
    for node in infected_nodes:
        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor]['status'] == Status.VULNERABLE.value:
                graph.nodes[neighbor]['status'] = Status.INFECTED.value
                new_infected_nodes.append(neighbor)
                logger.debug("SPREAD VIRUS: Node " + f'{neighbor}' + " has been infected from node " + f'{node}')
                #display_graph(graph)

    infected_nodes.clear()
    for node in new_infected_nodes:
        infected_nodes.append(node)  
    return bool(infected_nodes)


def spread_vaccination(graph:nx.DiGraph, vaccinated_nodes:list)->None:
    """
    Spread the vaccination from vaccinated nodes to their neighbors.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    vaccinated_nodes : list
        List of currently vaccinated nodes.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([(1, {"status": 'directly_vaccinated'}), (2, {"status": 'vulnerable'}), (3, {"status": 'vulnerable'})])
    >>> G.add_edges_from([(1, 2), (2, 3)])
    >>> vaccinated_nodes = [1]
    >>> spread_vaccination(G, vaccinated_nodes)
    >>> G.nodes(data=True)
    NodeDataView({1: {'status': 'directly_vaccinated'}, 2: {'status': 'vaccinated'}, 3: {'status': 'vulnerable'}})
    """
    new_vaccinated_nodes = []
    for node in vaccinated_nodes:
        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor]['status'] == Status.VULNERABLE.value:
                graph.nodes[neighbor]['status'] = Status.VACCINATED.value
                new_vaccinated_nodes.append(neighbor)
                logger.debug("SPREAD VACCINATION: Node " + f'{neighbor}' + " has been vaccinated from node " + f'{node}')
                #display_graph(graph)
    vaccinated_nodes.clear()
    for node in new_vaccinated_nodes:
        vaccinated_nodes.append(node)
    return

def vaccinate_node(graph:nx.DiGraph, node:int)->None:
    """
    Directly vaccinate a specific node in the graph.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    node : int
        Node to be vaccinated.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_node(1, status=0)
    >>> vaccinate_node(G, 1)
    >>> G.nodes(data=True)
    NodeDataView({1: {'status': 'directly vaccinated'}})
    """
    graph.nodes[node]['status'] = Status.DIRECTLY_VACCINATED.value
    logger.info("Node " + f'{node}' + " has been directly vaccinated")
    #display_graph(graph)
    return

def clean_graph(graph:nx.DiGraph)->None:
    """
    Reset the graph to its base state where all nodes are targets.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([(1, {"status": 1}), (2, {"status": 2}), (3, {"status": 0})])
    >>> clean_graph(G)
    >>> G.nodes(data=True)
    NodeDataView({1: {'status': 'vulnerable'}, 2: {'status': 'vulnerable'}, 3: {'status': 'vulnerable'}})
    """
    for node in graph.nodes:
        graph.nodes[node]['status'] = Status.VULNERABLE.value
    return

# ============================ End Spreading Max-Save ============================

# ===========================  Non-Spreading Min-Budget-Dirlay ============================
def adjust_nodes_capacity(graph:nx.DiGraph, source:int)->list:
    """
    Adjust the capacity of nodes based on the layer they belong to.
    The capacity is based on the formula in the article at the DirLayNet algorithm section.

    Parameters:
    ----------
    graph : nx.DiGraph
        Directed graph.
    source : int
        Source node.

    Returns:
    -------
    list
        List of nodes grouped by layers.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    >>> layers = adjust_nodes_capacity(G, 1)
    >>> layers
    [[1], [2], [3], [4]]
    >>> G.nodes(data=True)
    NodeDataView({1: {}, 2: {'capacity': 0.5454545454545455}, 3: {'capacity': 0.27272727272727276}, 4: {'capacity': 0.18181818181818182}})
    """
    logger.debug(f"Starting to adjust node capacity for dirlay graph nodes...") 

    layers = (list(nx.bfs_layers(graph,source)))
    harmonic_sum = 0.0
    for i in range(1,len(layers)):
        harmonic_sum = harmonic_sum + 1/i
    for index in range(1,len(layers)):
        for node in layers[index]:
            graph.nodes[node]['capacity'] = 1/(index*harmonic_sum)
            logger.info(f"Added Capacity {1/(index*harmonic_sum)} for node: {node}") 

    logger.info(f"Done with adding capacity for nodes, with Layers: {layers}")       

    return layers

def create_st_graph(graph:nx.DiGraph, targets:list)->nx.DiGraph:
    """
    Create an s-t graph from the original graph for use in connectivity algorithms.

    Parameters:
    ----------
    graph : nx.DiGraph
        Original directed graph.
    targets : list
        List of target nodes.

    Returns:
    -------
    nx.DiGraph
        s-t graph.

    Examples:
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    >>> targets = [2, 3]
    >>> G_st = create_st_graph(G, targets)
    >>> 't' in G_st.nodes
    True
    >>> list(G_st.successors(2))
    [3, 't']
    >>> list(G_st.successors(3))
    [4, 't']
    """
    logger.info(f"Creating a s-t graph to connect nodes to save") 

    G = copy.deepcopy(graph)
    G.add_node('t', status = Status.VULNERABLE.value)
    for node in targets:
        G.add_edge(node,'t')
    #display_graph(G)

    logger.info(f"Done creating a s-t graph") 
    return G

def min_cut_N_groups(graph: nx.DiGraph, source: int, layers: list) -> dict:
    """
    Find the minimum cut and group nodes accordingly.

    Parameters:
    ----------
    graph : nx.DiGraph
        Graph after flow reduction.
    source : int
        Source node.
    layers : list
        List of lists, where each sublist contains nodes belonging to that layer.

    Returns:
    -------
    dict
        Dictionary with layers as keys and lists of nodes in the minimum cut as values.

    Examples:
    --------

    """
    # Compute the minimum cut
    logger.info(f"Finding the minimum cut on the graph after reduction") 
    min_cut_nodes = algo.minimum_st_node_cut(graph, f'{source}_out', 't_in')
    
    logger.info(f"Minimum Cut is: {min_cut_nodes}")  

    # Initialize the groups dictionary with empty lists for each layer index
    groups = {i+1: [] for i in range(len(layers)-1)}
    logger.info(f"Finding the correct nodes from each layer according to the min-cut nodes") 
    
    # Populate the groups dictionary
    for item in min_cut_nodes:
        node , suffix = item.split('_')
        node = int(node)
        for i, layer_nodes in enumerate(layers):
            if node in layer_nodes:
                groups[i].append(node)
                break

    logger.info(f"Ni groups: {groups}") 
    return groups


def calculate_vaccine_matrix(layers:list, min_cut_nodes_grouped:dict)->np.matrix: 
    """
    Calculate the vaccine matrix based on the calculation in the article at the DirLayNet algorithm section.

    Parameters:
    ----------
    layers : list
        List of nodes grouped by layers.
    min_cut_nodes_grouped : dict
        List of nodes in the minimum cut grouped into layers.

    Returns:
    -------
    np.matrix
        Vaccine matrix.

    Examples:
    --------
    
    """
    logger.info(f"Calculating the Vaccine Matrix...")

    matrix_length = max(min_cut_nodes_grouped.keys()) 
    matrix = np.zeros((matrix_length, matrix_length))
    for j in range(matrix_length):
        for i in range(j+1):
                N_j = len(min_cut_nodes_grouped[j+1])
                value = N_j / (j + 1) 
                matrix[i][j] = value

    logger.info(f"Vaccination Matrix Before roundups:\n{matrix}")
    return matrix

def matrix_to_integers_values(matrix: np.matrix) -> np.matrix:
    """
    Convert a matrix with fractional entries to an integral matrix such that
    the row and column sums are either the floor or ceiling of the original sums.
    The solution is provided with a construction of a flow graph and then applying a max-flow algorithm on it.

    Parameters:
    ----------
    matrix : np.matrix
        The input matrix with fractional entries.

    Returns:
    -------
    np.matrix
        The converted integral matrix.

    Examples:
    --------
    
    """
    # dimensions of the matrix
    logger.info(f"Applying max-flow to transfer the Vaccine Matrix for integers...")
    rows, cols = matrix.shape
    
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    col_sums = np.array(matrix.sum(axis=0)).flatten()
    
    # logger.info(f"Row sums: {row_sums}")
    # logger.info(f"Column sums: {col_sums}")
    
    G = nx.DiGraph()
    
    # add source and sink nodes
    source = 's'
    sink = 't'
    G.add_node(source)
    G.add_node(sink)
    
    # add nodes for rows and columns
    row_nodes = ['r{}'.format(i) for i in range(rows)]
    col_nodes = ['c{}'.format(j) for j in range(cols)]
    G.add_nodes_from(row_nodes)
    G.add_nodes_from(col_nodes)
    
    # add edges from source to row nodes with capacities as the ceiling of row sums
    for i in range(rows):
        G.add_edge(source, row_nodes[i], capacity=np.ceil(row_sums[i]))
    
    # add edges from column nodes to sink with capacities as the ceiling of column sums
    for j in range(cols):
        G.add_edge(col_nodes[j], sink, capacity=np.ceil(col_sums[j]))
    
    # add edges from row nodes to column nodes with capacity 1
    for i in range(rows):
        for j in range(cols):
            G.add_edge(row_nodes[i], col_nodes[j], capacity=1)
    
    # computes the maximum flow
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    
    # builds the integral matrix
    integral_matrix = np.zeros_like(matrix, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if flow_dict[row_nodes[i]][col_nodes[j]] > 0:
                integral_matrix[i, j] = np.ceil(matrix[i, j])
            else:
                integral_matrix[i, j] = np.floor(matrix[i, j])

    logger.info(f"Integral and final Matrix:\n{integral_matrix}")
    
    return np.matrix(integral_matrix)


def min_budget_calculation(matrix: np.matrix) -> int:
    """
    Calculate the minimum budget from the matrix.

    Parameters:
    ----------
    matrix : np.matrix
        Input matrix.

    Returns:
    -------
    int
        Minimum budget.

    Examples:
    --------
    >>> matrix = np.matrix([[0.5, 1.5], [1.2, 0.8]])
    >>> min_budget_calculation(matrix)
    3
    """
    integral_matrix = matrix_to_integers_values(matrix)
    rows_sum = integral_matrix.sum(axis=1) # we get column sum as we want to -> on time step i, vaccinate Mij nodes from layer j , for all i ≤ j ≤ .
    min_budget = int(rows_sum.max())
    logger.info(f"Min budget needed to save the target nodes: {min_budget}")
    return min_budget

# ===========================  End Non-Spreading Max-Save ============================

# ===========================  Heuristic Utilities ===================================
def find_best_neighbor(graph:nx.DiGraph, infected_nodes:list, targets:list)->int:
    """
    Find the best node from the infected nodes successors that saves more new node in targets.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - infected_nodes (list): list of all infected nodes that threaten to infect additional nodes.
    - targets (list): List of target nodes.

    Returns:
    - best_node (int): Best node option.
    """
    best_node = None
    nodes_saved = []
    max_number = -1
    optional_nodes = set()

    # Go through the infected_nodes list and collect all their neighbors
    for node in infected_nodes:
        optional_nodes.update(graph.neighbors(node))

    for node in optional_nodes:
        if graph.nodes[node]['status'] == Status.VULNERABLE.value:
            # for each node that is target, we will add only his nighbors that are target as well
            neighbors_list = list(graph.neighbors(node))
            vulnerable_neighbors = set()
            for neighbor in neighbors_list:
                if graph.nodes[neighbor]['status'] == Status.VULNERABLE.value:
                    vulnerable_neighbors.add(neighbor)
            if node in targets:
                vulnerable_neighbors.add(node)
            common_elements = set(vulnerable_neighbors) & set(targets)
            logger.info("node " + f'{node}' + " is saving the nodes " + str(common_elements))
            if len(common_elements) > max_number:
                best_node = node
                nodes_saved = common_elements
                max_number = len(common_elements)

    if best_node != None:
     logger.info("The best node is: " + f'{best_node}' + " and it's saves nodes: " + str(nodes_saved))
    return best_node, nodes_saved
# ===========================  End Heuristic Utilities ================================

# ===========================  General Utilities ======================================
def display_graph(graph:nx.DiGraph)->None:
    """
    Display the graph using Matplotlib.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    """
    pos = nx.shell_layout(graph)
    colors = [node_colors.get(data.get('status', 'default'), 'default') for node, data in graph.nodes(data=True)]
    nx.draw(graph, pos, node_color=colors, with_labels=True, font_weight='bold')
    
    if nx.get_edge_attributes(graph, 'weight'):
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()
    return

def parse_json_to_networkx(json_data):
    """
    Parse JSON data to create Networkx graphs.

    Parameters:
    - json_data (dict): JSON data containing graph information.

    Returns:
    - graphs (dict): Dictionary of Networkx graphs.
    """
    graphs = {}
    for graph_type, graph_data in json_data.items():
        for graph_name, graph_info in graph_data.items():
            graph_key = f"{graph_type}_{graph_name}"
            
            if "vertices" not in graph_info or not isinstance(graph_info["vertices"], list) or not graph_info["vertices"]:
                logger.critical(f"Error parsing {graph_type}_{graph_name}: 'vertices' must be a non-empty list.")
                raise KeyError(f"Error parsing {graph_type}_{graph_name}: 'vertices' must be a non-empty list.")
            
            if "edges" not in graph_info or not isinstance(graph_info["edges"], list) or not graph_info["edges"]:
                logger.critical(f"Error parsing {graph_type}_{graph_name}: 'edges' must be a non-empty list.")
                raise KeyError(f"Error parsing {graph_type}_{graph_name}: 'edges' must be a non-empty list.")
            
            vertices = graph_info["vertices"]
            edges = [(edge[0], edge[1]) for edge in graph_info["edges"]]
            G = nx.DiGraph()
            G.add_nodes_from(vertices, status=Status.VULNERABLE.value)
            G.add_edges_from(edges)
            graphs[graph_key] = G
    return graphs

# ===========================  End General Utilities ============================

if __name__ == "__main__":
    import doctest
    result = doctest.testmod(verbose=True)
    logger.info(f"Doctest results: {result}")