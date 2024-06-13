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

setup_global_logger()

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import doctest
    result = doctest.testmod(firefighter_problem, verbose=True)
    logger.info(f"Doctest results: {result}")

    #G3 = nx.DiGraph() 
    #G3.add_nodes_from([0,1,2,3,4,5,6,7,8], status="vulnerable")
    #G3.add_edges_from([(0,2),(0,4),(0,5),(2,1),(2,3),(4,1),(4,6),(5,3),(5,6),(5,7),(6,7),(6,8),(7,8)])
    #logger.info("=" * 150)
    #logger.info(spreading_minbudget(G3,0,[2,6,1,8]))
    #print(spreading_maxsave(G3,1, 0,[2,6,1,8])[1])
    # logger.info("=" * 150)
    # logger.info(heuristic_minbudget(G3,0,[2,6,1,8], True))

