import experiments_csv
# add the comparisons test from both heuristic files + add a function that will plot the graphs. 
# TBD 

import logging
import json
import time
logger = logging.getLogger(__name__)


from networkz.algorithms.approximation.firefighter_problem.Utils import *
from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import *
from networkz.algorithms.approximation.tests.test_firefighter_problem.test_non_spreading_dirlaynet_minbudget import generate_layered_network

def setup_global_logger(level: int = logging.DEBUG):
    log_format = "|| %(asctime)s || %(levelname)s || %(message)s"
    date_format = '%H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


if __name__ == "__main__":
    setup_global_logger(level=logging.DEBUG)

    with open("networkz/algorithms/approximation/tests/test_firefighter_problem/graphs.json", "r") as file:
        json_data = json.load(file)
    graphs = parse_json_to_networkx(json_data)

    G_dirlay_random = generate_layered_network() #random graph generator for dirlay testings/ can also fit other algorithms but dirlay
    
    ex = experiments_csv.Experiment("results/", "non_spreading.csv", backup_folder=None)
    # ex.clear_previous_results() # to clear previous experminets..

    logger.debug("\nNon-Spread-Dirlay:\n")
    input_ranges = {
        "Graph":G_dirlay_random,
        "source": [0],
        "targets":[2,4],
    }
    ex.run(non_spreading_dirlaynet_minbudget,input_ranges)

    # ##

    # logger.debug("\nNon-Spread-MinBudget:\n")
    # input_ranges = {
    #     "Graph":G_dirlay_random,
    #     "source": 0,
    #     "targets":[2,4],
    # }
    # ex.run(non_spreading_minbudget,input_ranges)

    # ##

    # logger.debug("\nNon-Spread-Heuristic-MinBudget:\n")
    # input_ranges = {
    #     "Graph":G_dirlay_random,
    #     "source": 0,
    #     "targets":[2,4],
    #     "spreading":False
    # }
    # ex.run(heuristic_minbudget,input_ranges)

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # logger.debug("\Spread-Minbudget:\n")
    # input_ranges = {
    #     "Graph": G_dirlay_random,
    #     "source": 0,
    #     "targets": [2,4],
    # }
    # ex.run(spreading_minbudget,input_ranges)

    # ##

    # logger.debug("\Spread-MaxSave:\n")
    # input_ranges = {
    #     "Graph": G_dirlay_random,
    #     "budget": 1,
    #     "source": 0,
    #     "targets": [2,4],
    # }
    # ex.run(spreading_maxsave,input_ranges)

    # ##

    # logger.debug("\Spread-Heuristic-MinBudget:\n")
    # input_ranges = {
    #     "Graph":G_dirlay_random,
    #     "source": 0,
    #     "targets":[2,4],
    #     "spreading":True
    # }
    # ex.run(heuristic_minbudget,input_ranges)

    # ##

    # logger.debug("\Spread-Heuristic-MaxSave:\n")
    # input_ranges = {
    #     "Graph":G_dirlay_random,
    #     "budget": 1,
    #     "source": 0,
    #     "targets":[2,4],
    # }
    # ex.run(heuristic_maxsave,input_ranges)

    print("\n DataFrame: \n", ex.dataFrame)