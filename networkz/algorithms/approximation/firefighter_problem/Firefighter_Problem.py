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
import math
import logging
from networkz.algorithms.max_flow_with_node_capacity import min_cut_with_node_capacity
from networkz.algorithms.approximation.firefighter_problem.Utils import *

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger

logger = logging.getLogger('firefighter_problem_main')


def spreading_maxsave(Graph:nx.DiGraph, budget:int, source:int, targets:list, stop_condition=None) -> tuple[list, set]:    
    """
    Approximate the firefighter problem by maximizing the number of saved nodes.

    Parameters:
    ----------
    Graph : nx.DiGraph
        Directed graph representing the network.
    budget : int
        Number of nodes that can be vaccinated at each time step.
    source : int
        Source node where the infection starts.
    targets : list
        List of target nodes to be saved.
    stop_condition : optional
        If set, stops early if all targets are saved or if any target is infected.

    Returns:
    -------
    tuple:
        vaccination_strategy : list
            List of tuples representing the vaccination strategy.
        saved_target_nodes : set
            Set of nodes that are saved from infection.

    Examples:
    --------
    Example 1:
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([0,1,2], status="vulnerable")
    >>> G.add_edges_from([(0,1),(0,2),(1,2)])
    >>> spreading_maxsave(G,1,0,[1,2])
    ([(1, 1)], {1})

    Example 2:
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> spreading_maxsave(G1,1,0,[1,2,3])
    ([(1, 1)], {1, 3})

    Example 3:
    >>> G2 = nx.DiGraph()
    >>> G2.add_nodes_from([0,1,2,3,4,5,6], status="vulnerable")
    >>> G2.add_edges_from([(0,1),(0,2),(1,2),(1,4),(2,3),(2,6),(3,5)])
    >>> spreading_maxsave(G2,1,0,[1,2,3,4,5,6])
    ([(2, 1), (4, 2)], {2, 3, 4, 5, 6})

    Example 4:
    >>> G3 = nx.DiGraph() 
    >>> G3.add_nodes_from([0,1,2,3,4,5,6,7,8], status="vulnerable")
    >>> G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    >>> spreading_maxsave(G3,2,0,[1,2,3,4,5,6,7,8])
    ([(5, 1), (2, 1), (8, 2)], {1, 2, 3, 5, 6, 7, 8})
    """

    if budget < 1:
        logger.critical("Error: The budget must be at least 1")
        raise ValueError("Error: The budget must be at least 1")
        
    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the spreading_maxsave function with source node {source}, budget {budget}, and targets: {targets}")
    
    clean_graph(Graph)
    local_targets = targets.copy()
    infected_nodes = []
    vaccinated_nodes = []
    vaccination_strategy = []
    saved_target_nodes = set()
    can_spread = True
    Graph.nodes[source]['status'] = Status.INFECTED.value
    infected_nodes.append(source)
    
    logger.info("Calculating all possible direct vaccinations in each timestamp")
    gamma, direct_vaccinations = calculate_gamma(Graph, source, targets)
    
    logger.info("Calculating direct vaccinations grouping by timestamp")
    epsilon = calculate_epsilon(direct_vaccinations)
    
    time_step = 0
    while can_spread and time_step < len(epsilon):
        spread_vaccination(Graph, vaccinated_nodes)
        for i in range(budget):
            logger.info(f"Calculating the best direct vaccination strategy for the current time step that saves more new nodes in targets (Current budget: {i+1} out of {budget})")
            vaccination, nodes_saved = find_best_direct_vaccination(Graph, direct_vaccinations, epsilon[time_step], local_targets)
            
            if vaccination != ():
                logger.info(f"Found {vaccination} as a solution for current timestamp. Appending to vaccination strategy and vaccinating the node")
                vaccination_strategy.append(vaccination)
                
                chosen_node = vaccination[0]
                vaccinate_node(Graph, chosen_node)
                
                vaccinated_nodes.append(chosen_node)
                logger.info(f"Updated list of currently vaccinated nodes: {vaccinated_nodes}")

                if nodes_saved is not None:
                    saved_target_nodes.update(nodes_saved)
                    local_targets[:] = [element for element in local_targets if element not in nodes_saved]
                    logger.info(f"Updated list of targets: {local_targets}")

            else:
                logger.info(f"All nodes are either vaccinated or infected")
                break

        can_spread = spread_virus(Graph, infected_nodes)

        if stop_condition is not None:
            if len(local_targets) == 0 or any(node in infected_nodes for node in local_targets):
                logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saved the nodes: {saved_target_nodes}")
                return vaccination_strategy, saved_target_nodes

        time_step += 1

    logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saves the nodes: {saved_target_nodes}")
    return vaccination_strategy, saved_target_nodes


def spreading_minbudget(Graph:nx.DiGraph, source:int, targets:list) -> tuple[int, list]:
    """
    Approximate the firefighter problem by minimizing the budget required to save all target nodes.

    Parameters:
    ----------
    Graph : nx.DiGraph
        Directed graph representing the network.
    source : int
        Source node where the infection starts.
    targets : list
        List of target nodes to be saved.

    Returns:
    -------
    tuple:
        min_budget : int
            Minimum budget required to save all target nodes.
        best_strategy : list
            List of tuples representing the vaccination strategy.

    Examples:
    --------
    Example 1: 
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> spreading_minbudget(G1,0,[1,2,3])
    (2, [(1, 1), (2, 1)])

    Example 2: 
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> spreading_minbudget(G1,0,[1,3])
    (1, [(1, 1)])

    Example 3:
    >>> G2 = nx.DiGraph()
    >>> G2.add_nodes_from([0,1,2,3,4,5,6], status="vulnerable")
    >>> G2.add_edges_from([(0,1),(0,2),(1,2),(1,4),(2,3),(2,6),(3,5)])
    >>> spreading_minbudget(G2,0,[1,2,3,4,5,6])
    (2, [(2, 1), (1, 1)])

    Example 4:
    >>> G3 = nx.DiGraph() 
    >>> G3.add_nodes_from([0,1,2,3,4,5,6,7,8], status="vulnerable")
    >>> G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    >>> spreading_minbudget(G3,0,[1,2,3,4,5,6,7,8])
    (3, [(5, 1), (2, 1), (4, 1)])
    """

    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the spreading_minbudget function with source node {source} and targets: {targets}")

    min_value = 1
    max_value = len(targets)
    min_budget = max_value
    middle = math.floor((min_value + max_value) / 2)

    best_strategy = spreading_maxsave(Graph, min_budget, source, targets, True)[0]

    while min_value < max_value:
        logger.info(f"Calling maxsave with parameters - Source: {source}, Targets: {targets}, Budget: {middle}")
        strategy, nodes_saved = spreading_maxsave(Graph, middle, source, targets, True)
        
        common_elements = set(nodes_saved) & set(targets)

        if len(common_elements) == len(targets):
            logger.info(f"The current budget {middle} has saved all the targets!")
            max_value = middle
            if middle < min_budget:
                min_budget = middle
                best_strategy = strategy
        else:
            logger.info(f"The current budget {middle} didn't save all the targets!")
            min_value = middle + 1

        middle = math.floor((min_value + max_value) / 2)

    logger.info(f"Returning minimum budget: {middle} and the vaccination strategy: {best_strategy}")
    return min_budget, best_strategy

    
def non_spreading_minbudget(Graph:nx.DiGraph, source:int, targets:list) -> int:
    """
    Calculate the minimum budget required to save all target nodes in a non-spreading scenario.

    Parameters:
    ----------
    Graph : nx.DiGraph
        Directed graph representing the network.
    source : int
        Source node where the infection starts.
    targets : list
        List of target nodes to be saved.

    Returns:
    -------
    min_budget : int
        Minimum budget required to save all target nodes.

    Examples:
    --------
    Example 1: 
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> non_spreading_minbudget(G1,0,[1,3])
    (2, [(1, 1), (3, 1)])

    Example 2: 
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> non_spreading_minbudget(G1,0,[1,2,3])
    (2, [(1, 1), (2, 1)])

    Example 3:
    >>> G2 = nx.DiGraph()
    >>> G2.add_nodes_from([0,1,2,3,4,5,6], status="vulnerable")
    >>> G2.add_edges_from([(0,1),(0,2),(1,2),(1,4),(2,3),(2,6),(3,5)])
    >>> non_spreading_minbudget(G2,0,[1,2,3,4,5,6])
    (2, [(1, 1), (2, 1)])

    Example 4:
    >>> G3 = nx.DiGraph() 
    >>> G3.add_nodes_from([0,1,2,3,4,5,6,7,8], status="vulnerable")
    >>> G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    >>> non_spreading_minbudget(G3,0,[2,6,1,8])
    (3, [(2, 1), (4, 1), (5, 1)])
    """
    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the non_spreading_minbudget function with source node {source} and targets: {targets}")

    G = create_st_graph(Graph, targets, 't')
    min_cut = algo.minimum_st_node_cut(G, source, 't')
    logger.info(f"Minimum s-t node cut: {min_cut}")
    min_budget = len(min_cut)
    strategy = [(item, 1) for item in min_cut]

    logger.info(f"Returning minimum budget: {min_budget}")
    return min_budget, strategy

def non_spreading_dirlaynet_minbudget(Graph:nx.DiGraph, source:int, targets:list) -> tuple[int, list]:
    """
    Calculate the minimum budget required to save all target nodes in a non-spreading directed layer network.

    Parameters:
    ----------
    Graph : nx.DiGraph
        Directed graph representing the network.
    source : int
        Source node where the infection starts.
    targets : list
        List of target nodes to be saved.

    Returns:
    -------
    tuple:
        min_budget : int
            Minimum budget required to save all target nodes.
        best_strategy : list
            List of tuples representing the vaccination strategy.

    Examples:
    --------
    Example 1:
    >>> G4 = nx.DiGraph()
    >>> G4.add_nodes_from([0,1,2,3,4,5], status="vulnerable")
    >>> G4.add_edges_from([(0,1),(0,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5),(4,5)])
    >>> non_spreading_dirlaynet_minbudget(G4,0,[1,2,3,4,5])
    (2, {0: [2, 1]})
    """

    validate_parameters(Graph, source, targets)
    if not is_dag(Graph):
       logger.error("The graph is not a DAG graph, thus cannot run the algorithm")
       return
    
    #display_graph(Graph)
    logger.info(f"Starting the non_spreading_dirlaynet_minbudget function with source node {source} and targets: {targets}")

    layers = adjust_nodes_capacity(Graph, source)
    G = create_st_graph(Graph, targets, 't')
    #display_graph(G)
    G_reduction_min_cut = min_cut_with_node_capacity(G, source=source, target='t')
    N_groups = min_cut_N_groups(G_reduction_min_cut,layers)
    vacc_matrix = calculate_vaccine_matrix(layers, N_groups)
    integer_matrix = matrix_to_integers_values(vacc_matrix)
    min_budget = min_budget_calculation(vacc_matrix)
    strategy = dirlay_vaccination_startegy(integer_matrix, N_groups)

    logger.info(f"Returning minimum budget: {min_budget} and the vaccination strategy: {strategy}")
    return min_budget, strategy

def heuristic_maxsave(Graph:nx.DiGraph, budget:int, source:int, targets:list, spreading=True,  stop_condition=None) -> tuple[list, set]:
    """
    Approximate the firefighter problem by maximizing the number of saved nodes in our heuristic apprach.
    The heuristic approach is based on the local search problem. 
    
    Parameters
    ----------
    Graph : nx.DiGraph
        Directed graph representing the network.
    budget : int
        Number of nodes that can be vaccinated at each time step.
    source : int
        Source node where the infection starts.
    targets : list
        List of target nodes to be saved.
    spreading : bool
        If True, vaccination spreads to neighboring nodes.
    stop_condition : optional
        If set, stops early if all targets are saved or if any target is infected.
    
    Returns
    -------
    tuple:
        vaccination_strategy : list
            List of tuples representing the vaccination strategy.
        saved_target_nodes : set
            Set of saved target nodes.
        
    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([0, 1, 2, 3], status="vulnerable")
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> heuristic_maxsave(G, 1, 0, [1, 2, 3])
    ([(1, 1)], {1, 3})
    """
    
    if budget < 1:
        logger.critical("Error: The budget must be at least 1")
        raise ValueError("Error: The budget must be at least 1")

    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the heuristic_maxsave function with source node {source}, budget {budget}, targets: {targets}, and spreading: {spreading}")

    clean_graph(Graph)
    #display_graph(Graph)
    local_targets = targets.copy()
    infected_nodes = []
    vaccinated_nodes = []
    vaccination_strategy = []
    saved_target_nodes = set()
    can_spread = True
    Graph.nodes[source]['status'] = Status.INFECTED.value
    infected_nodes.append(source)
    #display_graph(Graph)
    time_step = 1

    while can_spread:
        if spreading:
            spread_vaccination(Graph, vaccinated_nodes)
        for i in range(budget):
            logger.info(f"Calculating the best direct vaccination strategy for the current time step that saves more new node in targets (Current budget: {budget})")
            node_to_vaccinate, nodes_saved = find_best_neighbor(Graph, infected_nodes, local_targets, targets)
            if node_to_vaccinate is not None:
                logger.info(f"Found {node_to_vaccinate} as a solution for current timestamp, appending to vaccination strategy and vaccinating the node")
                vaccination_strategy.append((node_to_vaccinate, time_step))
                vaccinate_node(Graph, node_to_vaccinate)
                vaccinated_nodes.append(node_to_vaccinate)
                logger.info(f"Updated list of currently vaccinated nodes: {vaccinated_nodes}")

                if nodes_saved is not None:
                    local_targets[:] = [element for element in local_targets if element not in nodes_saved]
                    logger.info(f"Updated list of targets: {local_targets}")

            else:
                logger.info(f"All nodes are either vaccinated or infected")
                break
        
        can_spread = spread_virus(Graph, infected_nodes)

        if stop_condition is not None:
            if len(local_targets) == 0 or any(node in infected_nodes for node in local_targets):
                for node in targets:
                    node_status = Graph.nodes[node]['status']
                    if node_status != Status.INFECTED.value:
                        saved_target_nodes.add(node)
                logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saved the nodes: {saved_target_nodes}")
                return vaccination_strategy, saved_target_nodes

        time_step += 1
    
    for node in targets:
        node_status = Graph.nodes[node]['status']
        if node_status != Status.INFECTED.value:
            saved_target_nodes.add(node)

    logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saves the nodes: {saved_target_nodes}")
    return vaccination_strategy, saved_target_nodes

def heuristic_minbudget(Graph:nx.DiGraph, source:int, targets:list, spreading:bool) -> tuple[int, list]:
    """
    Calculate the minimum budget required to save all target nodes in our heuristic approach.

   Parameters
    ----------
    Graph : nx.DiGraph
        Directed graph representing the network.
    source : int
        Source node where the infection starts.
    targets : list
        List of target nodes to be saved.
    spreading : bool
        If True, vaccination spreads to neighboring nodes.
    
    Returns
    -------
    tuple:
        min_budget : int
            Minimum budget required to save all target nodes.
        best_strategy : list
            List of tuples representing the vaccination strategy.
    
    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([0, 1, 2, 3], status="vulnerable")
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> heuristic_minbudget(G, 0, [1, 2, 3], True)
    (2, [(1, 1), (2, 1)])
    """
    
    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the heuristic_minbudget function with source node {source}, targets: {targets}, and spreading: {spreading}")

    min_value = 1
    max_value = len(list(Graph.successors(source)))
    min_budget = max_value
    middle = math.floor((min_value + max_value) / 2)

    best_strategy = heuristic_maxsave(Graph, min_budget, source, targets, spreading, True)[0]

    while min_value < max_value:
        logger.info(f"Calling heuristic_maxsave with parameters - Source: {source}, Targets: {targets}, Budget: {middle}")
        strategy, nodes_saved = heuristic_maxsave(Graph, middle, source, targets, spreading, True)

        common_elements = set(nodes_saved) & set(targets)

        if len(common_elements) == len(targets):
            logger.info(f"The current budget {middle} has saved all the targets!")
            max_value = middle
            if middle < min_budget:
                min_budget = middle
                best_strategy = strategy
        else:
            logger.info(f"The current budget {middle} didn't save all the targets!")
            min_value = middle + 1

        middle = math.floor((min_value + max_value) / 2)

    logger.info(f"Returning minimum budget: {middle} and the vaccination strategy: {best_strategy}")
    
    return min_budget, best_strategy

if __name__ == "__main__":

    import doctest
    result = doctest.testmod(verbose=False)
    print(f"Doctest results: {result}")