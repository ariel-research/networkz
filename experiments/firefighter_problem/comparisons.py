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
from networkz.algorithms.approximation.firefighter_problem.Random_Graph_Generator import generate_random_layered_network
from networkz.algorithms.approximation.firefighter_problem.Random_Graph_Generator import generate_random_DiGraph

logger = logging.getLogger(__name__)

def setup_global_logger(level: int = logging.DEBUG):
    """
    Setup the global logger with a specific format and logging level.

    Parameters:
    ----------
    level : int
        Logging level, e.g., logging.DEBUG, logging.INFO.
    """
    log_format = "|| %(asctime)s || %(levelname)s || %(message)s"
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

def runner_no_spreading(algorithm, graph, source, targets):
    """
    Run the specified algorithm without spreading.

    Parameters:
    ----------
    algorithm : function
        The algorithm to be executed.
    graph : nx.Graph
        The graph on which the algorithm is to be run.
    source : int
        The source node.
    targets : list
        The target nodes to be saved.

    Returns:
    -------
    dict:
        A dictionary containing the budget used by the algorithm.
    """
    if algorithm == heuristic_minbudget:
        result = algorithm(Graph=graph, source=source, targets=targets, spreading=False)
        return {"Budget": result}
    else:
        result = algorithm(Graph=graph, source=source, targets=targets)
        return {"Budget": result}

def runner_spreading(algorithm, graph, budget, source, targets):
    """
    Run the specified algorithm with spreading.

    Parameters:
    ----------
    algorithm : function
        The algorithm to be executed.
    graph : nx.Graph
        The graph on which the algorithm is to be run.
    source : int
        The source node.
    targets : list
        The target nodes to be saved.

    Returns:
    -------
    dict:
        A dictionary containing the budget used or nodes saved by the algorithm.
    """
    if algorithm == heuristic_minbudget:
        return {"Budget": (algorithm(Graph=graph, source=source, targets=targets, spreading=True))}
    
    if algorithm == heuristic_maxsave:
        return {"Nodes_Saved": (str(len(algorithm(Graph=graph, budget=budget, source=source, targets=targets, spreading=True)[1])))}
    
    if algorithm == spreading_maxsave:
        return {"Nodes_Saved": (str(len(algorithm(Graph=graph, budget=budget, source=source, targets=targets)[1])))}
    
    else:
        return {"Budget": (algorithm(Graph=graph, source=source, targets=targets))}

def Compare_NonSpread():
    """
    Compare the performance of different algorithms without spreading.

    This function runs multiple experiments on randomly generated layered networks
    and plots the results comparing the budget used by different algorithms.
    """
    ex1 = experiments_csv.Experiment("./experiments/", "non_spreading.csv", backup_folder=None)
    ex1.clear_previous_results()  # to clear previous experiments

    input_ranges = {
        "algorithm": [non_spreading_dirlaynet_minbudget, non_spreading_minbudget, heuristic_minbudget],
    }

    def multiple_runs(runs=30):
        for _ in range(runs):
            graph = generate_random_layered_network() 
            source = 0
            nodes = list(graph.nodes)
            nodes.remove(0)
            num_targets = random.randint(1, int(len(nodes) / 4) + 1)
            targets = random.sample(nodes, num_targets)
            for algorithm in input_ranges["algorithm"]:
                start_time = perf_counter()
                result = runner_no_spreading(algorithm, graph, source, targets)
                runtime = perf_counter() - start_time
                ex1.add({**{"algorithm": algorithm.__name__, "runtime": runtime, "graph_nodes": len(graph.nodes)}, **result})
        return {"status": "completed"}

    # Set a time limit for the entire batch run
    ex1.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

    # Preprocess the DataFrame to extract numeric budget values
    results_csv_file = "./experiments/non_spreading_minbudget.csv"
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
    preprocessed_csv_file = "./experiments/non_spreading_minbudget_preprocessed.csv"
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
        save_to_file="./experiments/non_spreading.png"
    )
    
    print("\n DataFrame-NonSpread: \n", ex1.dataFrame)

def Compare_SpreadingMaxSave():
    """
    Compare the performance of different algorithms with spreading for maximizing saved nodes.

    This function runs multiple experiments on randomly generated directed graphs
    and plots the results comparing the number of nodes saved by different algorithms.
    """
    ex2 = experiments_csv.Experiment("./experiments/", "spreading_maxsave.csv", backup_folder=None)
    ex2.clear_previous_results()  # to clear previous experiments

    input_ranges = {
        "algorithm": [spreading_maxsave, heuristic_maxsave]
    }
    
    node_counts = [100, 200, 400]
    edge_probabilities = [0.1, 0.5, 0.8]
    budget_range = [1,2,3,5,7,10]

    def multiple_runs(runs=10):
        for num_nodes in node_counts:
            for edge_prob in edge_probabilities:
                graph = generate_random_DiGraph(num_nodes=num_nodes, edge_probability=edge_prob, seed=None)
                for _ in range(runs):
                    source = 0
                    nodes = list(graph.nodes)
                    nodes.remove(0)
                    num_targets = random.randint(1, int(len(nodes) / 2) + 1)
                    targets = random.sample(nodes, num_targets)
                    for budget in budget_range:
                        for algorithm in input_ranges["algorithm"]:
                            start_time = perf_counter()
                            result = runner_spreading(algorithm, graph, budget, source, targets)
                            runtime = perf_counter() - start_time
                            
                            ex2.add({**{"algorithm": algorithm.__name__, "runtime": runtime, "Budget": budget, "graph_nodes": num_nodes, "edge_probability": edge_prob}, **result})
        return {"status": "completed"}

    # Set a time limit for the entire batch run
    ex2.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

    ## DATA ISSUE WE HAD SO THIS IS A FIX ##
    # Load the results
    results_csv_file = "./experiments/spreading_maxsave.csv"
    results = pd.read_csv(results_csv_file)

    # Ensure 'algorithm' column is of type string
    results['algorithm'] = results['algorithm'].astype(str)

    # Ensure 'Budget' column is numeric and drop rows with NaNs
    results['Budget'] = pd.to_numeric(results['Budget'], errors='coerce')
    results = results.dropna(subset=['Budget'])

    # Ensure 'Budget' is an integer
    results['Budget'] = results['Budget'].astype(int)

    # Ensure 'Nodes_Saved' column is numeric and drop rows with NaNs
    results['Nodes_Saved'] = pd.to_numeric(results['Nodes_Saved'], errors='coerce')
    results = results.dropna(subset=['Nodes_Saved'])

    # Ensure 'Nodes_Saved' is an integer
    results['Nodes_Saved'] = results['Nodes_Saved'].astype(int)

    # Save the cleaned DataFrame to a new CSV file (optional, for debugging)
    cleaned_csv_file = "./experiments/spreading_maxsave_preprocessed.csv"
    results.to_csv(cleaned_csv_file, index=False)

    # Plot the results using the cleaned DataFrame
    multi_plot_results(
        results_csv_file=cleaned_csv_file,
        filter={}, 
        subplot_rows=2,
        subplot_cols=3,
        x_field="graph_nodes", 
        y_field="Nodes_Saved", 
        z_field="algorithm", 
        subplot_field="Budget",
        sharex=True,
        sharey=True,
        mean=True,
        save_to_file="./experiments/spreading_maxsave_budget.png"
    )

    multi_plot_results(
        results_csv_file=cleaned_csv_file,
        filter={"graph_nodes":100}, 
        subplot_rows=3,
        subplot_cols=1,
        x_field="Budget", 
        y_field="Nodes_Saved", 
        z_field="algorithm", 
        subplot_field="edge_probability",
        sharex=True,
        sharey=True,
        mean=True,
        save_to_file="./experiments/spreading_maxsave_100_edge_prob.png"
    )

    multi_plot_results(
        results_csv_file=cleaned_csv_file,
        filter={"graph_nodes":200}, 
        subplot_rows=3,
        subplot_cols=1,
        x_field="Budget", 
        y_field="Nodes_Saved", 
        z_field="algorithm", 
        subplot_field="edge_probability",
        sharex=True,
        sharey=True,
        mean=True,
        save_to_file="./experiments/spreading_maxsave_200_edge_prob.png"
    )

    multi_plot_results(
        results_csv_file=cleaned_csv_file,
        filter={"graph_nodes":400}, 
        subplot_rows=3,
        subplot_cols=1,
        x_field="Budget", 
        y_field="Nodes_Saved", 
        z_field="algorithm", 
        subplot_field="edge_probability",
        sharex=True,
        sharey=True,
        mean=True,
        save_to_file="./experiments/spreading_maxsave_400_edge_prob.png"
    )

    print("\n DataFrame-NonSpread: \n", ex2.dataFrame)

def Compare_SpreadingMinBudget():
    """
    Compare the performance of different algorithms with spreading for minimizing the budget.

    This function runs multiple experiments on randomly generated directed graphs
    and plots the results comparing the budget used by different algorithms.
    """
    ex3 = experiments_csv.Experiment("./experiments/", "spreading_minbudget.csv", backup_folder=None)
    ex3.clear_previous_results()  # to clear previous experiments

    input_ranges = {
        "algorithm": [spreading_minbudget, heuristic_minbudget]
    }
    
    node_counts = [100, 200, 400]
    edge_probabilities = [0.1, 0.5, 0.8]

    def multiple_runs(runs=10):
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
                        result = runner_spreading(algorithm, graph, None, source, targets)
                        runtime = perf_counter() - start_time
                        ex3.add({**{"algorithm": algorithm.__name__, "runtime": runtime, "graph_nodes": num_nodes, "edge_probability": edge_prob}, **result})
        return {"status": "completed"}

    # Set a time limit for the entire batch run
    ex3.run_with_time_limit(multiple_runs, input_ranges={}, time_limit=0.9)

    # Preprocess the DataFrame to extract numeric budget values
    results_csv_file = "./experiments/spreading_minbudget.csv"
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
    preprocessed_csv_file = "./experiments/spreading_minbudget_preprocessed.csv"
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
        save_to_file="./experiments/spreading_minbudget.png"
    )

    single_plot_results(
        results_csv_file=preprocessed_csv_file,
        filter={"edge_probability":0.1}, 
        x_field="graph_nodes", 
        y_field="Budget_numeric", 
        z_field="algorithm", 
        mean=True,
        save_to_file="./experiments/spreading_minbudget_edge.png"
    )
    
    print("\n DataFrame-NonSpread: \n", ex3.dataFrame)

if __name__ == "__main__":
    """To run this - please run one at a time; mark the others and then run"""

    setup_global_logger(level=logging.DEBUG)

    #Compare_NonSpread()
    Compare_SpreadingMinBudget()
    #Compare_SpreadingMaxSave()