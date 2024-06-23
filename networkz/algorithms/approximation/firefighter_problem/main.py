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

import logging
import json
logger = logging.getLogger(__name__)


# This is a fix for an issue where the top one has to be exclusive for pytest to work
# and the bottom one needs to be exclusive for running this from terminal to work
from networkz.algorithms.approximation.firefighter_problem.Utils import *
from networkz.algorithms.approximation.firefighter_problem.Firefighter_Problem import *
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


if __name__ == "__main__":
    import doctest
    setup_global_logger(level=logging.DEBUG)

    # result = doctest.testmod(firefighter_problem, verbose=True)
    # logger.info(f"Doctest results: {result}")

    # G3 = nx.DiGraph() 
    # G3.add_nodes_from([0,1,2,3,4,5,6,7,8])
    # G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    # logger.info("=" * 150)
    #print(spreading_maxsave(G3,source=0,targets=[2,6,1,8],budget=1))
    #print(spreading_minbudget(G3,source=0,targets=[2,6,1,8]))

    with open("networkz/algorithms/tests/test_firefighter_problem/graphs.json", "r") as file:
        json_data = json.load(file)
    graphs = parse_json_to_networkx(json_data)

    G2 = graphs["Dirlay_Graph-2"]
    print(non_spreading_dirlaynet_minbudget(Graph=G2, src=0, targets=[2,4])) 

    # G2 = graphs["RegularGraph_Graph-2"]
    # print(heuristic_minbudget(G2,source=0, targets=[1,3,4,5,6],spreading=True))
    #print(spreading_maxsave(G3,1, 0,[2,6,1,8])[1])
    # print(spreading_minbudget(G2,source=0, targets=[1,3,4,5,6]))
    # logger.info("=" * 150)
    #logger.info(heuristic_minbudget(G3,0,[2,6,1,8], True))

