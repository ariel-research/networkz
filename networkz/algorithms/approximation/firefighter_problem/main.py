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

import logging
import json
import time
logger = logging.getLogger(__name__)


# This is a fix for an issue where the top one has to be exclusive for pytest to work
# and the bottom one needs to be exclusive for running this from terminal to work
from networkz.algorithms.approximation.firefighter_problem.Utils import *
from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import *
from networkz.algorithms.approximation.tests.test_firefighter_problem.test_non_spreading_dirlaynet_minbudget import generate_layered_network
import networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem as firefighter_problem # to run the doctest on the firefighter_problem files

def setup_global_logger(level: int = logging.DEBUG):
    log_format = "|| %(asctime)s || %(levelname)s || %(message)s"
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

def compare_functions(graph: nx.DiGraph, source: int):
    """
    This method comapres the function adjust_nodes_capacity 
    with a more complex version of it that uses threadpool to calcualte the capacity of each node
    """
    graph1 = graph.copy()
    graph2 = graph.copy()

    start_time = time.time()
    adjust_nodes_capacity(graph1, source)
    sequential_time = time.time() - start_time

    start_time = time.time()
    adjust_nodes_capacity_parallel(graph2, source)
    parallel_time = time.time() - start_time

    # Plot the execution times
    plt.figure(figsize=(10, 5))
    
    times = [sequential_time, parallel_time]
    labels = ['adjust_nodes_capacity', 'adjust_nodes_capacity_parallel']
    
    plt.bar(labels, times, color=['b', 'g'], alpha=0.6)
    plt.title('Sequential vs Parallel Execution Times')
    plt.xlabel('Execution Type')
    plt.ylabel('Time (seconds)')
    plt.show()


def compare_spreading_algorithms(graph: nx.DiGraph, source: int, targets:list):
    """
    This method is used to compare the algoritmhs we created and show thier running time.
    """
    graph1 = graph.copy()
    graph2 = graph.copy()
    graph3 = graph.copy()
    graph4 = graph.copy()
   

    start_time1 = time.time()
    heuristic_maxsave(graph1, 1,source, targets)
    end_time1 = time.time() - start_time1

    start_time2 = time.time()
    heuristic_minbudget(graph2,source, targets, True) #spreading 
    end_time2 = time.time() - start_time2

    start_time3 = time.time()
    spreading_minbudget(graph3,source, targets)
    end_time3 = time.time() - start_time3

    start_time4 = time.time()
    spreading_maxsave(graph4, 1,source, targets)
    end_time4 = time.time() - start_time4

    # Plot the execution times
    plt.figure(figsize=(10, 5))
    
    times = [end_time1, end_time2, end_time3 ,end_time4]
    labels = ['heuristic_maxsave','heuristic_minbudget', 'minbudget', 'maxsave']
    
    plt.bar(labels, times, color=['b', 'b', 'g', 'g'], alpha=0.6)
    plt.title('Algorithm Running time comparison')
    plt.xlabel('Execution Type')
    plt.ylabel('Time (seconds)')
    plt.show()

def compare_non_spreading_algorithm(graph: nx.DiGraph, source: int, targets:list):
    """
    This method is used to compare the algoritmhs we created and show thier running time.
    """
    graph1 = graph.copy()
    graph2 = graph.copy()
    graph3 = graph.copy()

    start_time1 = time.time()
    non_spreading_dirlaynet_minbudget(graph1, source,targets)
    end_time1= time.time() - start_time1

    start_time2 = time.time()
    heuristic_minbudget(graph2,source, targets, False) #no spreading
    end_time2 = time.time() - start_time2

    start_time3 = time.time()
    non_spreading_minbudget(graph3,source, targets)
    end_time3 = time.time() - start_time3

    # Plot the execution times
    plt.figure(figsize=(10, 5))
    
    times = [end_time1, end_time2, end_time3]
    labels = ['dirlaynet_minbudget', 'heuristic_minbudget', 'mincut_minbudget']
    
    plt.bar(labels, times, color=['b', 'g', 'y'], alpha=0.6)
    plt.title('Non spreading algorithms Running time comparison')
    plt.xlabel('Execution Type')
    plt.ylabel('Time (seconds)')
    plt.show()

if __name__ == "__main__":
    import doctest
    setup_global_logger(level=logging.DEBUG)

    with open("networkz/algorithms/tests/test_firefighter_problem/graphs.json", "r") as file:
        json_data = json.load(file)
    graphs = parse_json_to_networkx(json_data)

    G_dirlay_random = generate_layered_network() #random graph generator for dirlay testings/ can also fit other algorithms but dirlay

    "Simple testings on spreading algorithms:" 
    # G2 = graphs["RegularGraph_Graph-2"]
    # print(heuristic_minbudget(G2,source=0, targets=[1,3,4,5,6],spreading=True))
    #print(spreading_maxsave(G3,1, 0,[2,6,1,8])[1])
    # print(spreading_minbudget(G2,source=0, targets=[1,3,4,5,6]))
    # logger.info("=" * 150)
    #logger.info(heuristic_minbudget(G3,0,[2,6,1,8], True))

    # G3 = nx.DiGraph() 
    # G3.add_nodes_from([0,1,2,3,4,5,6,7,8])
    # G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    # logger.info("=" * 150)
    #print(spreading_maxsave(G3,source=0,targets=[2,6,1,8],budget=1))
    #print(spreading_minbudget(G3,source=0,targets=[2,6,1,8]))

    "Dirlay simple running exmaple:"
    G2 = graphs["Dirlay_Graph-2"]
    # print(non_spreading_dirlaynet_minbudget(Graph=G2, src=0, targets=[2,4])) 

    "Comapring adjust_nodes_capacity with parallel running:"
    # compare_functions(G_dirlay_random,0) # more complex random graph comparison

    "Compare Graph Algorithms:"
    # compare_algorithms(G_dirlay_random,0, [2,4]) # more complex random graph comparison
    compare_non_spreading_algorithm(G_dirlay_random, 0, [2,4]) # simple graph example comparison
    compare_spreading_algorithms(G_dirlay_random, 0, [2,4]) 


