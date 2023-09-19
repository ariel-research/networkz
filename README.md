# NetworkZ

NetworkZ is a library of graph algorithms in Python. It is an extension of the [NetworkX](https://github.com/networkx/networkx). It contains (by import) everything that is in NetworkX, plus some additional algorithms that were submitted into NetworkX but not merged yet. Currently, NetworkZ contains the following additional algorithms:

* [Rank-maximal matching](networkz/algorithms/bipartite/rank_maximal_matching.py)
* [Approximate coalition formation](networkz/algorithms/approximation/coalition_formation.py)

## Quick Example
### Rank Maximal Matching Algorithm
```
import networkz as nx
G = nx.Graph()
G.add_nodes_from(["a1", "a2"], bipartite=0)
G.add_nodes_from(["p1", "p2"], bipartite=1)
G.add_weighted_edges_from([("a1", "p2", 1), ("a1", "p1", 1), ("a2", "p2", 2)])
matching = nx.rank_maximal_matching(G, rank="weight")
print(matching)
```

## Contribution
New additions are welcome! \
For essential guidance on contributing, consult the [NetworkX Contributor Guide](https://github.com/networkx/networkx/blob/main/CONTRIBUTING.rst).
