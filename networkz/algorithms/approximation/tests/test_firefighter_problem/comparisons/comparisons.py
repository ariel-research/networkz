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

import experiments_csv
from experiments_csv import *
import logging
logger = logging.getLogger(__name__)


from networkz.algorithms.approximation.firefighter_problem.Utils import *
from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import *
from networkz.algorithms.approximation.tests.test_firefighter_problem.test_non_spreading_dirlaynet_minbudget import generate_layered_network
from matplotlib import pyplot as plt

def setup_global_logger(level: int = logging.DEBUG):
    log_format = "|| %(asctime)s || %(levelname)s || %(message)s"
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

G_dirlay_random = generate_layered_network()

def runner_no_spreading(algorithm):
    graph = G_dirlay_random.copy()
    source = 0
    targets = [2,4,6,7,8,9,20,15]

    if algorithm == heuristic_minbudget:
         return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets,spreading=False))}
    
    else:
        return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets))}

        
def runner_spreading(algorithm):
    graph = G_dirlay_random.copy()
    source = 0
    targets = [2,4,6,7,8,9,20,15]

    if algorithm == heuristic_minbudget:
         return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets,spreading=True))}
    
    if algorithm == heuristic_maxsave:
        return {"Budget" : (algorithm(Graph=graph,budget = 1, source=source,targets=targets,spreading=True))}
    
    if algorithm == spreading_maxsave:
        return {"Budget" : (algorithm(Graph=graph,budget = 1, source=source,targets=targets))}
    
    else:
        return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets))}


if __name__ == "__main__":
    setup_global_logger(level=logging.DEBUG)


    G_dirlay_random = generate_layered_network() #random graph generator for dirlay testings/ can also fit other algorithms but dirlay
    
    ex1 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "non_spreading.csv", backup_folder=None)
    ex1.clear_previous_results() # to clear previous experminets..

    input_ranges = {
        "algorithm":[non_spreading_dirlaynet_minbudget,non_spreading_minbudget,heuristic_minbudget],
    }
    ex1.run_with_time_limit(runner_no_spreading,input_ranges, time_limit=0.9)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    ex2 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "spreading.csv", backup_folder=None)
    ex2.clear_previous_results() # to clear previous experminets..

    input_ranges = {
        "algorithm":[spreading_minbudget,spreading_maxsave,heuristic_minbudget,heuristic_maxsave],
    }
    ex2.run_with_time_limit(runner_spreading,input_ranges, time_limit=0.9)


    #Plotting:

    single_plot_results("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/non_spreading.csv", 
                        filter = {}, 
                        x_field="algorithm", 
                        y_field="runtime", 
                        z_field="Budget", 
                        save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/non_spreading.png")
    
    print("\n DataFrame-NonSpread: \n", ex1.dataFrame)


    single_plot_results("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading.csv", 
                        filter = {}, 
                        x_field="algorithm", 
                        y_field="runtime", 
                        z_field="Budget", 
                        save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading.png")

    print("\n DataFrame-Spreading: \n", ex2.dataFrame)