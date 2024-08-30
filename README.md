# NetworkZ

NetworkZ is a library of graph algorithms in Python. It is an extension of the [NetworkX](https://github.com/networkx/networkx). It contains (by import) everything that is in NetworkX, plus some additional algorithms that were submitted into NetworkX but not merged yet. Currently, NetworkZ contains the following additional algorithms:

* [Rank-maximal matching](networkz/algorithms/bipartite/rank_maximal_matching.py): by Oriya Alperin, Liel Vaknin and Amiel Lejzor.
* [Maximum-weight fractional matching](networkz/algorithms/max_weight_fractional_matching.py): by Oriya Alperin, Liel Vaknin and Amiel Lejzor.
* [Social-aware coalition formation](networkz/algorithms/approximation/coalition_formation.py) - by Victor Kushnir.
* [Minimum cut on a graph with node capacity](networkz/algorithms/max_flow_with_node_capacity.py): by Yuval Bubnovsky, Almog David and Shaked Levi.
* [Several approximate solutions to the Firefighter problem](networkz/algorithms/approximation/firefighter_problem): by Yuval Bubnovsky, Almog David and Shaked Levi.

## Installation

```
pip install networkz
```

This installs the latest version of networkx, and the new algorithms added in networkz.


## Usage

### Rank Maximal Matching
A rank-maximal matching is a matching that maximizes the number of agents who are matched to their 1st priority; subject to that, it maximizes the number of agents matched to their 2nd priority; and so on.

```python
import networkz as nx
G = nx.Graph()
G.add_nodes_from(["agent1", "agent2"], bipartite=0)
G.add_nodes_from(["product1", "product2"], bipartite=1)
G.add_weighted_edges_from([("agent1", "product1", 1), ("agent1", "product2", 1), ("agent2", "product2", 2)])
matching = nx.rank_maximal_matching(G, rank="weight")
print(matching)
```

See [demo website](https://rmm.csariel.xyz/) for more information.


### Maximum-Weight Fractional Matching
Maximum-weight fractional matching is a graph optimization problem where the goal is to find a set of edges with maximum total weight, allowing for fractional inclusion of edges.

```python
import networkz as nx
G = nx.Graph()
G.add_nodes_from(["a1", "a2"])
G.add_edge("a1", "a2", weight=3)
F = nx.maximum_weight_fractional_matching(G)
print(F)
```

### Approximating The Fire-Fighter Problem
The Firefighter problem models the case where a diffusive process such as an infection (or an idea, a computer virus, a fire) is spreading through a network, and our goal is to contain this infection by using targeted vaccinations.

Networkz implements several algorithms to approximate solutions for the fire-fighter problems in 2 models: spreading & non-spreading, where the virus/fire always spreads but the spread of vaccination is dependant on the model.

Under each such model, we are intrested in two problem types: MaxSave (save as many nodes from the target list given a budget) and MinBudget (What is the minimum budget needed to save all of the node target list)

```python
import networkz as nx
G = nx.DiGraph()
G.add_nodes_from([0,1,2,3,4,5,6], status="vulnerable")
G.add_edges_from([(0,1),(0,2),(1,2),(1,4),(2,3),(2,6),(3,5)])

strategy, saved_nodes = nx.spreading_maxsave(G, budget = 1, source = 0, targets = [1,2,3,4,5,6]) 
# Will return ([(2, 1), (4, 2)], {2, 3, 4, 5, 6})

min_budget, strategy = nx.spreading_minbudget(G2, source = 0,targets = [1,2,3,4,5,6]) 
min_budget, strategy = nx.non_spreading_minbudget(G, source = 0,targets = [1,2,3,4,5,6])
# Both will return (2, [(1, 1), (2, 1)])
```

Another algorithm which is implemented in networkz is MinBudget in a non-spreading model (vaccine doesn't spread), running on a directed-layered network graph:
```python
import networkz as nx
G = nx.DiGraph()
G.add_nodes_from([0,1,2,3,4,5], status="vulnerable")
G.add_edges_from([(0,1),(0,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5),(4,5)])

# The algorithm will check if this is actually a directed-layered network graph
min_budget, strategy = nx.non_spreading_dirlaynet_minbudget(G, source = 0,targets = [1,2,3,4,5])
# Will return (2, [(1, 1), (2, 1)])
```

See the [algorithms in actions](https://the-firefighters.github.io/WebsiteGit/) for more information and a virtual sandbox to run and observe the algirithms.

### Social-Aware Coalition Formation

(TODO)


## Contribution

Any additions or bug-fixes to `networkx` should first be submitted there, according to the [NetworkX Contributor Guide](https://github.com/networkx/networkx/blob/main/CONTRIBUTING.rst).

If the pull-request is not handled, you are welcome to submit it here too.





