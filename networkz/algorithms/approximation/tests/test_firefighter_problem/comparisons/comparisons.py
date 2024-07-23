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
from time import perf_counter
import pandas as pd
import random

from networkz.algorithms.approximation.firefighter_problem.Utils import *
from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import *
from networkz.algorithms.approximation.tests.test_firefighter_problem.test_non_spreading_dirlaynet_minbudget import generate_layered_network
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

def setup_global_logger(level: int = logging.DEBUG):
    log_format = "|| %(asctime)s || %(levelname)s || %(message)s"
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

def generate_random_DiGraph(
    num_nodes: int = 100,
    edge_probability: float = 0.1,
    seed: int = None
    ) -> nx.DiGraph:
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.debug(f"Random generated seed: {seed}")
    else:
        logger.debug(f"Using provided seed: {seed}")
    
    random.seed(seed)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes), status="target")
    
    edges = [
        (source, target) 
        for source in range(num_nodes) 
        for target in range(num_nodes) 
        if source != target and random.random() < edge_probability
    ]
    
    G.add_edges_from(edges)
    
    return G
    

def runner_no_spreading(algorithm, graph, source, targets):
    if algorithm == heuristic_minbudget:
        result = algorithm(Graph=graph, source=source, targets=targets, spreading=False)
        return {"Budget": result}
    else:
        result = algorithm(Graph=graph, source=source, targets=targets)
        return {"Budget": result}

        
def runner_spreading( algorithm, graph, source, targets):

    if algorithm == heuristic_minbudget:
         return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets,spreading=True))}
    
    if algorithm == heuristic_maxsave:
        return {"Nodes_Saved" : (str(len(algorithm(Graph=graph,budget = 2, source=source,targets=targets,spreading=True)[1])))}
    
    if algorithm == spreading_maxsave:
        return {"Nodes_Saved" : (str(len(algorithm(Graph=graph,budget = 2, source=source,targets=targets)[1])))}
    
    else:
        return {"Budget" : (algorithm(Graph=graph,source=source,targets=targets))}


def Compare_NonSpread():
    ex1 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "non_spreading.csv", backup_folder=None)
    ex1.clear_previous_results() # to clear previous experminets..

    input_ranges = {
        "algorithm":[non_spreading_dirlaynet_minbudget,non_spreading_minbudget,heuristic_minbudget],

    }
    def multiple_runs(runs=100):
        for _ in range(runs):
            graph = generate_layered_network()
            source = 0
            # targets = [2, 4, 6, 7, 8, 9]
            nodes = list(graph.nodes)
            nodes.remove(0)
            num_targets = random.randint(1, int(len(nodes)/4)+1)
            targets = random.sample(nodes,num_targets)
            for algorithm in input_ranges["algorithm"]:
                start_time = perf_counter()
                result = runner_no_spreading(algorithm, graph, source, targets)
                runtime = perf_counter() - start_time
                ex1.add({**{"algorithm": algorithm.__name__, "runtime": runtime, "graph_nodes": len(graph.nodes)}, **result})
        return {"status": "completed"}

    # Set a time limit for the entire batch run
    ex1.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

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
        mean=True,
        save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/non_spreading.png"
    )
    
    print("\n DataFrame-NonSpread: \n", ex1.dataFrame)

def Compare_SpreadingMaxSave():
    ex1 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "spreading_maxsave.csv", backup_folder=None)
    ex1.clear_previous_results()  # to clear previous experiments

    input_ranges = {
        "algorithm": [spreading_maxsave, heuristic_maxsave]
    }
    
    node_counts = [100, 200, 400]
    edge_probabilities = [0.1, 0.5, 0.8]

    def multiple_runs(runs=1):
        for num_nodes in node_counts:
            for edge_prob in edge_probabilities:
                graph = generate_random_DiGraph(num_nodes=num_nodes, edge_probability=edge_prob, seed=None)
                for _ in range(runs):
                    source = 0
                    nodes = list(graph.nodes)
                    nodes.remove(0)
                    num_targets = random.randint(1, int(len(nodes) / 2) + 1)
                    targets = random.sample(nodes, num_targets)
                    for algorithm in input_ranges["algorithm"]:
                        start_time = perf_counter()
                        result = runner_spreading(algorithm, graph, source, targets)
                        runtime = perf_counter() - start_time

                        if algorithm == spreading_maxsave:
                            id = 1
                        else:
                            id = 2 # heuristic
                        
                        ex1.add({**{"algorithm": id, "runtime": runtime, "graph_nodes": len(graph.nodes)}, **result})
        return {"status": "completed"}


    # Set a time limit for the entire batch run
    ex1.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

    # Plot the results using the preprocessed CSV file
    single_plot_results(
        results_csv_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading_maxsave.csv",
        filter={}, 
        x_field="graph_nodes", 
        y_field="Nodes_Saved", 
        z_field="algorithm", 
        mean=True,
        save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading_maxsave.png"
    )
    
    print("\n DataFrame-NonSpread: \n", ex1.dataFrame)

def Compare_SpreadingMinBudget():
    ex1 = experiments_csv.Experiment("./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/", "spreading_minbudget.csv", backup_folder=None)
    ex1.clear_previous_results()  # to clear previous experiments

    input_ranges = {
        "algorithm": [spreading_minbudget, heuristic_minbudget]
    }
    
    node_counts = [100, 200, 400]
    edge_probabilities = [0.1, 0.5, 0.8]

    def multiple_runs(runs=1):
        for num_nodes in node_counts:
            for edge_prob in edge_probabilities:
                graph = generate_random_DiGraph(num_nodes=num_nodes, edge_probability=edge_prob, seed=None)
                for _ in range(runs):
                    source = 0
                    nodes = list(graph.nodes)
                    nodes.remove(0)
                    num_targets = random.randint(1, int(len(nodes) / 2) + 1)
                    targets = random.sample(nodes, num_targets)
                    for algorithm in input_ranges["algorithm"]:
                        start_time = perf_counter()
                        result = runner_spreading(algorithm, graph, source, targets)
                        runtime = perf_counter() - start_time
                        ex1.add({**{"algorithm": algorithm.__name__, "runtime": runtime, "graph_nodes": len(graph.nodes)}, **result})
        return {"status": "completed"}


    # Set a time limit for the entire batch run
    ex1.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

    # Preprocess the DataFrame to extract numeric budget values
    results_csv_file = "./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading_minbudget.csv"
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
    preprocessed_csv_file = "./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading_minbudget_preprocessed.csv"
    results.to_csv(preprocessed_csv_file, index=False)

    print("\n DataFrame-NonSpread: \n", results)

    # Plot the results using the preprocessed CSV file
    single_plot_results(
        results_csv_file=preprocessed_csv_file,
        filter={}, 
        x_field="graph_nodes", 
        y_field="Budget_numeric", 
        z_field="algorithm", 
        mean=True,
        save_to_file="./networkz/algorithms/approximation/tests/test_firefighter_problem/comparisons/spreading_minbudget.png"
    )
    
    print("\n DataFrame-NonSpread: \n", ex1.dataFrame)

if __name__ == "__main__":
    setup_global_logger(level=logging.DEBUG)
    Compare_NonSpread()
    #Compare_SpreadingMinBudget()
    #Compare_SpreadingMaxSave()