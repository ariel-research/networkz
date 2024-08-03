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
import random
import networkx as nx

logger = logging.getLogger(__name__)

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


#TODO : make this to receive paramaters
def generate_random_layered_network():
    """
    Generates a directed layered network with a random number of layers and random nodes per layer.
    
    Returns:
        G (networkx.DiGraph): Directed graph representing the layered network.
    """
    # Randomly decide the number of layers (between 2 and 3 for this example)
    num_layers = random.randint(5, 10)
    
    # Randomly decide the number of nodes per layer (between 1 and 4 for this example)
    nodes_per_layer = [random.randint(5, 30) for _ in range(num_layers)]
    
    G = nx.DiGraph()
    node_id = 1  # Start node_id from 1 because 0 is the source
    
    # Initialize layer 0 with the source node
    layers = [[0]]
    G.add_node(0)
    
    # Create nodes layer by layer
    for i in range(num_layers):
        layer = [node_id + j for j in range(nodes_per_layer[i])]
        layers.append(layer)
        G.add_nodes_from(layer)
        node_id += nodes_per_layer[i]

    print("LAYERS->", layers)
    
    # Connect source node (0) to all nodes in layer 1
    for node in layers[1]:
        G.add_edge(0, node)
    
    # Create edges ensuring connectivity between consecutive layers
    for i in range(1, num_layers):
        for node in layers[i]:
            # Connect each node in this layer to at least one node in the next layer
            connected_nodes = random.sample(layers[i + 1], k=random.randint(1, len(layers[i + 1])))
            for target in connected_nodes:
                if target != node:  # Ensure no self-loop
                    G.add_edge(node, target)
        
        for target in layers[i + 1]:
            # Ensure each node in the next layer is connected to from at least one node in this layer
            if not any(G.has_edge(source, target) for source in layers[i]):
                G.add_edge(random.choice(layers[i]), target)
    
    return G
