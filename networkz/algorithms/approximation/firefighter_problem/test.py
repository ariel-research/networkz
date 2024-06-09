import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_random_directed_layered_graph(layers, nodes_per_layer, edge_probability):
    """
    Generates a random directed layered network graph.
    
    Parameters:
        layers (int): Number of layers in the graph.
        nodes_per_layer (int): Number of nodes per layer.
        edge_probability (float): Probability of creating an edge between nodes in consecutive layers.
        
    Returns:
        G (networkx.DiGraph): A directed layered network graph.
    """
    G = nx.DiGraph()
    
    # Create nodes for each layer
    for layer in range(layers):
        for node in range(nodes_per_layer):
            G.add_node(f"L{layer}_N{node}", layer=layer)
    
    # Create edges between nodes in consecutive layers based on edge probability
    for layer in range(layers - 1):
        for node1 in range(nodes_per_layer):
            for node2 in range(nodes_per_layer):
                if random.random() < edge_probability:
                    G.add_edge(f"L{layer}_N{node1}", f"L{layer + 1}_N{node2}")
    
    return G

def draw_layered_graph(G):
    """
    Draws the layered graph with layers arranged in horizontal layers.
    
    Parameters:
        G (networkx.DiGraph): A directed layered network graph.
    """
    pos = {}
    layer_nodes = {}
    
    for node, data in G.nodes(data=True):
        layer = data['layer']
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
    
    for layer, nodes in layer_nodes.items():
        for i, node in enumerate(nodes):
            pos[node] = (layer, -i)
    
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", arrowsize=20)
    plt.show()

# Example usage:
layers = 3
nodes_per_layer = 4
edge_probability = 0.3

G = generate_random_directed_layered_graph(layers, nodes_per_layer, edge_probability)
draw_layered_graph(G)