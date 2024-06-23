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
import math
import logging
from networkz.algorithms.approximation.firefighter_problem.graph_flow_reduction import max_flow_with_node_capacity
from networkz.algorithms.approximation.firefighter_problem.Utils import *

logger = logging.getLogger(__name__)

def spreading_maxsave(Graph:nx.DiGraph, budget:int, source:int, targets:list, stop_condition=None) -> list:
    """
    "Approximability of the Firefighter Problem - Computing Cuts over Time",
    by Elliot Anshelevich, Deeparnab Chakrabarty, Ameya Hate, Chaitanya Swamy (2010)
    https://link.springer.com/article/10.1007/s00453-010-9469-y
    
    Programmers: Shaked Levi, Almog David, Yuval Bubnovsky

    spreading_maxsave: Gets a directed graph, budget, source node, and list of targeted nodes that we need to save
    and return the best vaccination strategy that saves the most nodes from the targeted nodes list.
    
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

    infected_nodes = []
    vaccinated_nodes = []
    vaccination_strategy = []
    saved_target_nodes = set()
    can_spread = True
    Graph.nodes[source]['status'] = Status.INFECTED.value
    infected_nodes.append(source)
    
    logger.info("Calculating all possible direct vaccinations in each timestamp")
    gamma, direct_vaccinations = calculate_gamma(Graph, source, targets)
    
    logger.info("Calculating direct vaccinations groups by timestamp")
    epsilon = calculate_epsilon(direct_vaccinations)
    
    time_step = 0
    while can_spread and time_step < len(epsilon):
        spread_vaccination(Graph, vaccinated_nodes)
        for i in range(budget):
            logger.info(f"Calculating the best direct vaccination strategy for the current time step that saves more new node in targets (Current budget: {budget})")
            vaccination, nodes_saved = find_best_direct_vaccination(Graph, direct_vaccinations, epsilon[time_step], targets)
            
            if vaccination != ():
                logger.info(f"Found {vaccination} as a solution for current timestamp, appending to vaccination strategy and vaccinating the node")
                vaccination_strategy.append(vaccination)
                
                chosen_node = vaccination[0]
                vaccinate_node(Graph, chosen_node)
                
                vaccinated_nodes.append(chosen_node)
                logger.info(f"Updated list of currently vaccinated nodes: {vaccinated_nodes}")

                if nodes_saved is not None:
                    saved_target_nodes.update(nodes_saved)
                    targets[:] = [element for element in targets if element not in nodes_saved]
                    logger.info(f"Updated list of targets: {targets}")

            else:
                logger.info(f"All nodes are either vaccinated or infected")

        can_spread = spread_virus(Graph, infected_nodes)

        if stop_condition is not None:
            if len(targets) == 0 or any(node in infected_nodes for node in targets):
                clean_graph(Graph)
                logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saved the nodes: {saved_target_nodes}")
                return vaccination_strategy, saved_target_nodes

        time_step += 1

    logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saved the nodes: {saved_target_nodes}")
    return vaccination_strategy, saved_target_nodes


def spreading_minbudget(Graph:nx.DiGraph, source:int, targets:list)-> int:
    """
    "Approximability of the Firefighter Problem - Computing Cuts over Time",
    by Elliot Anshelevich, Deeparnab Chakrabarty, Ameya Hate, Chaitanya Swamy (2010)
    https://link.springer.com/article/10.1007/s00453-010-9469-y
    
    Programmers: Shaked Levi, Almog David, Yuval Bubnovsky

    spreading_minbudget: Gets a directed graph, source node, and list of targeted nodes that we need to save
    and returns the minimum budget that saves all the nodes from the targeted nodes list.

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

    original_targets = list(targets)
    best_strategy = []
    
    min_value = 1
    max_value = len(targets)
    middle = math.floor((min_value + max_value) / 2)

    while min_value < max_value:
        logger.info(f"Calling maxsave with parameters - Source: {source}, Targets: {targets}, Budget: {middle}")
        strategy, nodes_saved = spreading_maxsave(Graph, middle, source, targets, True)
        
        common_elements = set(nodes_saved) & set(original_targets)

        if len(common_elements) == len(original_targets):
            logger.info(f"The current budget {middle} has saved all the targets!")
            max_value = middle
            best_strategy = strategy
        else:
            logger.info(f"The current budget {middle} didn't save all the targets!")
            min_value = middle + 1

        middle = math.floor((min_value + max_value) / 2)
        targets = list(original_targets)

    logger.info(f"Returning minimum budget: {middle} and the vaccination strategy: {best_strategy}")
    return middle, best_strategy

    
def non_spreading_minbudget(Graph:nx.DiGraph, source:int, targets:list)->int:
    """
    "Approximability of the Firefighter Problem - Computing Cuts over Time",
    by Elliot Anshelevich, Deeparnab Chakrabarty, Ameya Hate, Chaitanya Swamy (2010)
    https://link.springer.com/article/10.1007/s00453-010-9469-y
    
    Programmers: Shaked Levi, Almog David, Yuval Bubnovsky

    non_spreading_minbudget: Gets a directed graph, source node, and list of targeted nodes that we need to save
    and returns the minimum budget that saves all the nodes from the targeted nodes list.
    
    Example1: 
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> non_spreading_minbudget(G1,0,[1,3])
    2

    Example 2: 
    >>> G1 = nx.DiGraph()
    >>> G1.add_nodes_from([0,1,2,3], status="vulnerable")
    >>> G1.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    >>> non_spreading_minbudget(G1,0,[1,2,3])
    2

    Example 3:
    >>> G2 = nx.DiGraph()
    >>> G2.add_nodes_from([0,1,2,3,4,5,6], status="vulnerable")
    >>> G2.add_edges_from([(0,1),(0,2),(1,2),(1,4),(2,3),(2,6),(3,5)])
    >>> non_spreading_minbudget(G2,0,[1,2,3,4,5,6])
    2

    Example 4:
    >>> G3 = nx.DiGraph() 
    >>> G3.add_nodes_from([0,1,2,3,4,5,6,7,8], status="vulnerable")
    >>> G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    >>> non_spreading_minbudget(G3,0,[2,6,1,8])
    3
    """
    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the non_spreading_minbudget function with source node {source} and targets: {targets}")

    G = create_st_graph(Graph, targets)
    min_budget = len(algo.minimum_st_node_cut(G, source, 't'))

    logger.info(f"Returning minimum budget: {min_budget}")
    return min_budget

def non_spreading_dirlaynet_minbudget(Graph:nx.DiGraph, src:int, targets:list)->int:
    """
    "Approximability of the Firefighter Problem - Computing Cuts over Time",
    by Elliot Anshelevich, Deeparnab Chakrabarty, Ameya Hate, Chaitanya Swamy (2010)
    https://link.springer.com/article/10.1007/s00453-010-9469-y
    
    Programmers: Shaked Levi, Almog David, Yuval Bubnovsky

    non_spreading_dirlaynet_minbudget: Gets a directed graph, source node, and list of targeted nodes that we need to save
    and returns the minimum budget that saves all the nodes from the targeted nodes list.
    
    Example1:
    >>> G4 = nx.DiGraph()
    >>> G4.add_nodes_from([0,1,2,3,4,5], status="vulnerable")
    >>> G4.add_edges_from([(0,1),(0,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5),(4,5)])
    >>> non_spreading_dirlaynet_minbudget(G4,0,[1,2,3,4,5])
    2
    """
    validate_parameters(Graph, src, targets)
    if not is_dag(Graph):
       logger.error("Problem with graph, its not a DAG, thus cannot run algorithm")
       return
    
    #display_graph(Graph)
    logger.info(f"Starting the non_spreading_dirlaynet_minbudget function with source node {src} and targets: {targets}")

    layers = adjust_nodes_capacity(Graph, src)
    G = create_st_graph(Graph, targets)
    #display_graph(G)
    G_reduction = max_flow_with_node_capacity(G, source=src, target='t')
    N_groups = min_cut_N_groups(G_reduction, src,layers)
    vacc_matrix = calculate_vaccine_matrix(layers, N_groups)
    min_budget = min_budget_calculation(vacc_matrix)

    logger.info(f"Returning algorithm stategy: {min_budget}")
    return min_budget

def heuristic_maxsave(Graph:nx.DiGraph, budget:int, source:int, targets:list, spreading=True,  stop_condition=None) -> tuple:
    """
    This heuristic approach is based on the local search problem. 
    We will select the best neighbor that saves the most nodes from targets.
    
    Parameters:
    - Graph (nx.DiGraph): Directed graph representing the network.
    - budget (int): Number of nodes that can be vaccinated at each time step.
    - source (int): Source node where the infection starts.
    - targets (list): List of target nodes to be saved.
    - spreading (bool): If True, vaccination spreads to neighboring nodes.
    - flag (bool, optional): If set, stops early if all targets are saved or if any target is infected.
    
    Returns:
    - list: List of tuples representing the vaccination strategy.
    
    Raises:
    - ValueError: If the budget is less than 1.
    
    Example:
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([0, 1, 2, 3], status="vulnerable")
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> heuristic_maxsave(G, 1, 0, [1, 2, 3])
    [(1, 1)]
    """
    if budget < 1:
        logger.critical("Error: The budget must be at least 1")
        raise ValueError("Error: The budget must be at least 1")

    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the heuristic_maxsave function with source node {source}, budget {budget}, targets: {targets}, and spreading: {spreading}")

    clean_graph(Graph)
    #display_graph(Graph)
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
            node_to_vaccinate, nodes_saved = find_best_neighbor(Graph, infected_nodes, targets)
            if node_to_vaccinate is not None:
                logger.info(f"Found {node_to_vaccinate} as a solution for current timestamp, appending to vaccination strategy and vaccinating the node")
                vaccination_strategy.append((node_to_vaccinate, time_step))
                vaccinate_node(Graph, node_to_vaccinate)
                vaccinated_nodes.append(node_to_vaccinate)
                logger.info(f"Updated list of currently vaccinated nodes: {vaccinated_nodes}")


                if nodes_saved is not None:
                    saved_target_nodes.update(nodes_saved)
                    targets[:] = [element for element in targets if element not in nodes_saved]
                    logger.info(f"Updated list of targets: {targets}")

            else:
                logger.info(f"All nodes are either vaccinated or infected")
        
        can_spread = spread_virus(Graph, infected_nodes)

        if stop_condition is not None:
            if len(targets) == 0 or any(node in infected_nodes for node in targets):
                logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saved the nodes: {saved_target_nodes}")
                return vaccination_strategy

        time_step += 1
    
    for node in targets:
        if Graph.nodes[node]['status'] != Status.INFECTED.value:
            saved_target_nodes.add(node)

    logger.info(f"Returning vaccination strategy: {vaccination_strategy}. The strategy saved the nodes: {saved_target_nodes}")
    return vaccination_strategy, saved_target_nodes

def heuristic_minbudget(Graph:nx.DiGraph, source:int, targets:list, spreading:bool)-> tuple:
    """
    This function calculates the minimum budget required to save all target nodes 
    using the heuristic approach based on local search problem.
    
    Parameters:
    - Graph (nx.DiGraph): Directed graph representing the network.
    - source (int): Source node where the infection starts.
    - targets (list): List of target nodes to be saved.
    - spreading (bool): If True, vaccination spreads to neighboring nodes.
    
    Returns:
    - int: Minimum budget required to save all target nodes.
    
    Example:
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from([0, 1, 2, 3], status="vulnerable")
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
    >>> heuristic_minbudget(G, 0, [1, 2, 3], True)
    (2, [(1, 1), (2, 1)])
    """
    validate_parameters(Graph, source, targets)
    logger.info(f"Starting the heuristic_minbudget function with source node {source}, targets: {targets}, and spreading: {spreading}")

    best_strategy= []
    original_targets = list(targets)
    min_value = 1
    max_value = len(targets)
    middle = math.floor((min_value + max_value) / 2)
    saved_everyone = True

    while min_value < max_value:
        strategy = heuristic_maxsave(Graph, middle, source, targets, spreading, True)

        for node in original_targets:
            if Graph.nodes[node]['status'] == Status.INFECTED.value:
                saved_everyone = False
                break

        if saved_everyone:
            logger.info(f"The current budget {middle} has saved all the targets!")
            max_value = middle
            best_strategy = strategy
        else:
            logger.info(f"The current budget {middle} didn't save all the targets!")
            min_value = middle + 1

        middle = math.floor((min_value + max_value) / 2)
        targets = list(original_targets)

    logger.info(f"Returning minimum budget: {middle} and the vaccination strategy: {best_strategy}")
    return middle, best_strategy

if __name__ == "__main__":
    import doctest
    result = doctest.testmod(verbose=False)
    logger.info(f"Doctest results: {result}")
    