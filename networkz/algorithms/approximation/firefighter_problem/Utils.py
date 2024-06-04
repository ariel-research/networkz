import networkx as nx
import networkx.algorithms.connectivity as algo 
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import logging

node_colors = {
    'target': 'gray',
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
    - graph (nx.DiGraph): Directed graph.
    - source (int): Source node.
    - targets (list): List of target nodes to save.

    Raises:
    - ValueError: If the source node is not in the graph.
    - ValueError: If the source node is in the targets list.
    - ValueError: If any target node is not in the graph.
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
    
"Spreading:"
def calculate_gamma(graph:nx.DiGraph, source:int, targets:list)-> dict:
    """
    Calculate Gamma and S(u,t)  based on the calculation in the article.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - source (int): Source node.
    - targets (list): List of target nodes to save.

    Returns:
    - gamma (dict): Dictionary of vaccination options.
    - direct_vaccination (dict): Dictionary of direct vaccination strategies - S(u,t).
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
    - direct_vaccinations (dict): Dictionary of direct vaccination strategies.

    Returns:
    - epsilon (list): List of direct vaccination groups by time step.
    """
    epsilon = []
    sorted_dict = dict(sorted(direct_vaccinations.items(), key=lambda item: item[0][1]))

    current_time_step = None
    current_group = []
    for key, value in sorted_dict.items():
        if current_time_step is None or key[1] == current_time_step:
            current_group.append(key)
        else:
            epsilon.append(current_group)
            current_group = [key]
        current_time_step = key[1]

    if current_group:
        epsilon.append(current_group)
    
    logger.info("Epsilon is: " + str(epsilon))
    return epsilon

def find_best_direct_vaccination(graph:nx.DiGraph, direct_vaccinations:dict, current_time_options:list, targets:list)->tuple:
    """
    Find the best direct vaccination strategy for the current time step that saves more new node in targets.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - direct_vaccinations (dict): Dictionary of direct vaccination strategies.
    - current_time_options (list): List of current time step vaccination options.
    - targets (list): List of target nodes.

    Returns:
    - best_vaccination (tuple): Best direct vaccination option.
    """
    best_vaccination = () 
    nodes_saved = {}
    common_elements = None
    max_number = -1
    for option in current_time_options:
        if(graph.nodes[option[0]]['status'] == 'target'):
            nodes_list = direct_vaccinations.get(option)
            common_elements = set(nodes_list) & set(targets)
            logger.debug(f"Direct vaccination: {option}, Nodes saved: {common_elements} (if set(), then it's empty)")
            if len(common_elements) > max_number:
                best_vaccination = option
                nodes_saved = common_elements
                max_number = len(common_elements)

    if nodes_saved is not None:
        targets[:] = [element for element in targets if element not in nodes_saved]
    
    if best_vaccination != ():
        logger.info("The best direct vaccination is: " + str(best_vaccination) + " and it saves nodes: " + str(nodes_saved))
    return best_vaccination

def spread_virus(graph:nx.DiGraph, infected_nodes:list)->bool:
    """
    Spread the virus from infected nodes to their neighbors.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - infected_nodes (list): List of currently infected nodes.

    Returns:
    - bool: True if there are new infections, False otherwise.
    """
    new_infected_nodes = []
    for node in infected_nodes:
        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor]['status'] == 'target':
                graph.nodes[neighbor]['status'] = 'infected'
                new_infected_nodes.append(neighbor)
                logger.debug("SPREAD VIRUS: Node " + f'{neighbor}' + " has been infected from node " + f'{node}')
    infected_nodes.clear()
    for node in new_infected_nodes:
        infected_nodes.append(node)  
    return bool(infected_nodes)


def spread_vaccination(graph:nx.DiGraph, vaccinated_nodes:list)->None:
    """
    Spread the vaccination from vaccinated nodes to their neighbors.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - vaccinated_nodes (list): List of currently vaccinated nodes.
    """
    new_vaccinated_nodes = []
    for node in vaccinated_nodes:
        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor]['status'] == 'target':
                graph.nodes[neighbor]['status'] = 'vaccinated'
                new_vaccinated_nodes.append(neighbor)
                logger.debug("SPREAD VACCINATION: Node " + f'{neighbor}' + " has been vaccinated from node " + f'{node}')
    vaccinated_nodes.clear()
    for node in new_vaccinated_nodes:
        vaccinated_nodes.append(node)
        logger.debug(f"Currently vaccinated nodes: {vaccinated_nodes}")
    return

def vaccinate_node(graph:nx.DiGraph, node:int)->None:
    """
    Directly vaccinate a specific node in the graph.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - node (int): Node to be vaccinated.
    """
    graph.nodes[node]['status'] = 'directly vaccinated'
    logger.info("Node " + f'{node}' + " has been directly vaccinated")
    return

def clean_graph(graph:nx.DiGraph)->None:
    """
    Reset the graph to its base state where all nodes are targets.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    """
    for node in graph.nodes:
        graph.nodes[node]['status'] = 'target'
    return

"Non-Spreading:"

def adjust_nodes_capacity(graph:nx.DiGraph, source:int)->list:
    """
    Adjust the capacity of nodes based on the layer they belong to.
    The capacity is based on the formula in the article at the DirLayNet algorithm section.

    Parameters:
    - graph (nx.DiGraph): Directed graph.
    - source (int): Source node.

    Returns:
    - layers (list): List of nodes grouped by layers.
    """
    layers = (list(nx.bfs_layers(graph,source)))
    harmonic_sum = 0.0
    for i in range(1,len(layers)):
        harmonic_sum = harmonic_sum + 1/i
    for index in range(1,len(layers)):
        for node in layers[index]:
            graph.nodes[node]['capacity'] = 1/(index*harmonic_sum)
    logger.info(f"Layers: {layers}")       
    return layers

def create_st_graph(graph:nx.DiGraph, targets:list)->nx.DiGraph:
    """
    Create an s-t graph from the original graph for use in connectivity algorithms.

    Parameters:
    - graph (nx.DiGraph): Original directed graph.
    - targets (list): List of target nodes.

    Returns:
    - G (nx.DiGraph): s-t graph.
    """
    G = copy.deepcopy(graph)
    G.add_node('t', status = 'target')
    for node in targets:
        G.add_edge(node,'t')
    #display_graph(G)
    return G

def graph_flow_reduction(graph:nx.DiGraph, source:int)->nx.DiGraph:
    """
    Perform flow reduction on the graph to find the minimum s-t cut.

    Parameters:
    - graph (nx.DiGraph): Original directed graph.
    - source (int): Source node.

    Returns:
    - H (nx.DiGraph): Graph after flow reduction.
    """
    H = nx.DiGraph()
    for node in graph.nodes:
        in_node, out_node = f'{node}_in', f'{node}_out'
        H.add_nodes_from([in_node, out_node])
        if node == source or node == 't':
            H.add_edge(in_node, out_node, weight=float('inf'))
        else:
            H.add_edge(in_node, out_node, weight=graph.nodes[node]['capacity'])
    for edge in graph.edges:
        H.add_edge(f'{edge[0]}_out', f'{edge[1]}_in', weight=float('inf'))
    # display_graph(H)
    return H

def min_cut_N_groups(graph: nx.DiGraph, source: int, layers: list) -> dict:
    """
    Find the minimum cut and group nodes accordingly.

    Parameters:
    - graph (nx.DiGraph): Graph after flow reduction.
    - source (int): Source node.
    - layers (list): List of lists, where each sublist contains nodes belonging to that layer.

    Returns:
    - groups (dict): Dictionary with layers as keys and lists of nodes in the minimum cut as values.
    """
    # Compute the minimum cut
    flow_graph = algo.minimum_st_node_cut(graph, f'{source}_out', 't_in')
    
    # Initialize the groups dictionary with empty lists for each layer index
    groups = {i+1: [] for i in range(len(layers)-1)}
    
    # Populate the groups dictionary
    for item in flow_graph:
        node , suffix = item.split('_')
        node = int(node)
        for i, layer_nodes in enumerate(layers):
            if node in layer_nodes:
                groups[i].append(node)
                break

    return groups


def calculate_vaccine_matrix(layers:list, min_cut_nodes_grouped:dict)->np.matrix: 
    """
    Calculate the vaccine matrix based on the calculation in the article at the DirLayNet algorithm section.

    Parameters:
    - layers (list): List of nodes grouped by layers.
    - min_cut_nodes_group (list): List of nodes in the minimum cut grouped into layers of them.

    Returns:
    - matrix (np.matrix): Vaccine matrix.
    """

    logger.info(f"Layers: {layers}")
    logger.info(f"Min cut nodes grouped: {min_cut_nodes_grouped}")
    matrix_length = max(min_cut_nodes_grouped.keys()) 
    matrix = np.zeros((matrix_length, matrix_length))
    for j in range(matrix_length):
        for i in range(j+1):
                N_j = len(min_cut_nodes_grouped[j+1])
                value = N_j / (j + 1) 
                matrix[i][j] = value
    logger.info(f"Matrix: {matrix}")
    return matrix

def matrix_to_integers_values(matrix: np.matrix) -> np.matrix:
    """
    Convert a matrix with fractional entries to an integral matrix such that
    the row and column sums are either the floor or ceiling of the original sums.
    The solution is provided with a consturction of a flow graph and then applying a max-flow algorihm on it.

    Parameters:
    matrix (np.matrix): The input matrix with fractional entries.

    Returns:
    np.matrix: The converted integral matrix.
    """
    # dimensions of the matrix
    rows, cols = matrix.shape
    
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    col_sums = np.array(matrix.sum(axis=0)).flatten()
    
    logger.info(f"Row sums: {row_sums}")
    logger.info(f"Column sums: {col_sums}")
    
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
    logger.info(f"Matrix: {integral_matrix}")
    return np.matrix(integral_matrix)


def min_budget_calculation(matrix: np.matrix) -> int:
    """
    Calculate the minimum budget from the matrix.

    Parameters:
    - matrix (np.matrix): Input matrix.

    Returns:
    - int: Minimum budget.
    """
    integral_matrix = matrix_to_integers_values(matrix)
    columns_sum = integral_matrix.sum(axis=0) # we get column sum as we want to -> on time step i, vaccinate Mij nodes from layer j , for all i ≤ j ≤ .
    min_budget = int(columns_sum.max())
    return min_budget


"Heuristic approach:"

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
        if graph.nodes[node]['status'] == 'target':
            # for each node that is target, we will add only his nighbors that are target as well
            neighbors_list = list(graph.neighbors(node))
            target_neighbors = set()
            for neighbor in neighbors_list:
                if graph.nodes[neighbor]['status'] == 'target':
                    target_neighbors.add(neighbor)
            if node in targets:
                target_neighbors.add(node)
            common_elements = set(target_neighbors) & set(targets)
            logger.info("node " + f'{node}' + " is saving the nodes " + str(common_elements))
            if len(common_elements) > max_number:
                best_node = node
                nodes_saved = common_elements
                max_number = len(common_elements)

    if nodes_saved is not None:
        targets[:] = [element for element in targets if element not in nodes_saved]

    if best_node != None:
     logger.info("The best node is: " + f'{best_node}' + " and it's saves nodes: " + str(nodes_saved))
    return best_node

"Usefull Utils:"

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
            edges = [(edge["source"], edge["target"]) for edge in graph_info["edges"]]
            G = nx.DiGraph()
            G.add_nodes_from(vertices, status="target")
            G.add_edges_from(edges)
            graphs[graph_key] = G
    return graphs