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


def runner_no_spreading(algorithm, graph, source, targets):
    if algorithm == heuristic_minbudget:
        result = algorithm(Graph=graph, source=source, targets=targets, spreading=False)
        return {"Budget": result}
    else:
        result = algorithm(Graph=graph, source=source, targets=targets)
        return {"Budget": result}

        
# def runner_spreading(algorithm, source):
#     graph = G_dirlay_random.copy()
#     targets = [2,4,6,7,8,9,20,15]

#     if algorithm == heuristic_minbudget:
#          return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets,spreading=True))}
    
#     if algorithm == heuristic_maxsave:
#         return {"Budget" : (algorithm(Graph=graph,budget = 1, source=source,targets=targets,spreading=True))}
    
#     if algorithm == spreading_maxsave:
#         return {"Budget" : (algorithm(Graph=graph,budget = 1, source=source,targets=targets))}
    
#     else:
#         return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets))}


from time import perf_counter
import pandas as pd
import random

if __name__ == "__main__":
    setup_global_logger(level=logging.DEBUG)


    G_dirlay_random = generate_layered_network() #random graph generator for dirlay testings/ can also fit other algorithms but dirlay
    
    ex1 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "non_spreading.csv", backup_folder=None)
    ex1.clear_previous_results() # to clear previous experminets..

    input_ranges = {
        "algorithm":[non_spreading_dirlaynet_minbudget,non_spreading_minbudget,heuristic_minbudget],

    }
    def multiple_runs(runs=50):
        for _ in range(runs):
            graph = generate_layered_network()
            source = 0
            # targets = [2, 4, 6, 7, 8, 9]
            nodes = list(graph.nodes)
            nodes.remove(0)
            num_targets = random.randint(1, int(len(nodes)/4))
            targets = random.sample(nodes,num_targets)
            for algorithm in input_ranges["algorithm"]:
                print(f"TEST LAYERS FOR ALGO {algorithm}, with graph -> {graph.nodes}")
                start_time = perf_counter()
                result = runner_no_spreading(algorithm, graph, source, targets)
                runtime = perf_counter() - start_time
                ex1.add({**{"algorithm": algorithm.__name__, "runtime": runtime, "graph_nodes": len(graph.nodes)}, **result})
        return {"status": "completed"}

    # Set a time limit for the entire batch run
    ex1.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

    # ex1.run_with_time_limit(runner_no_spreading,input_ranges, time_limit=0.9)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # ex2 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "spreading.csv", backup_folder=None)
    # ex2.clear_previous_results() # to clear previous experminets..

    # input_ranges = {
    #     "algorithm":[spreading_minbudget,spreading_maxsave,heuristic_minbudget,heuristic_maxsave],
    #     "source" : [0,4,6,7]
    # }
    # ex2.run_with_time_limit(runner_spreading,input_ranges, time_limit=0.9)


    #Plotting:

# Preprocess the DataFrame to extract numeric budget values
    results_csv_file = "./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/non_spreading.csv"
    results = pd.read_csv(results_csv_file)

    # Extract the numeric budget from the 'Budget' column
    def extract_budget_numeric(budget):
        if isinstance(budget, tuple):
            return budget[0]
        elif isinstance(budget, str):
            try:
                return eval(budget)[0]
            except:
                return None
        return None

    results['Budget_numeric'] = results['Budget'].apply(extract_budget_numeric)

    # Drop rows where the 'Budget_numeric' is not available
    results = results.dropna(subset=['Budget_numeric'])

    # Save the preprocessed DataFrame to a temporary CSV file
    preprocessed_csv_file = "./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/non_spreading_preprocessed.csv"
    results.to_csv(preprocessed_csv_file, index=False)

    print("\n DataFrame-NonSpread: \n", results)

    # Plot the results using the preprocessed CSV file
    single_plot_results(
        results_csv_file=preprocessed_csv_file,
        filter={}, 
        x_field="graph_nodes", 
        y_field="Budget_numeric", 
        z_field="algorithm", 
        save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/non_spreading.png"
    )
    
    print("\n DataFrame-NonSpread: \n", ex1.dataFrame)


    # single_plot_results("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading.csv", 
    #                     filter = {}, 
    #                     x_field="algorithm", 
    #                     y_field="runtime", 
    #                     z_field="Budget", 
    #                     save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading.png")

    # print("\n DataFrame-Spreading: \n", ex2.dataFrame)