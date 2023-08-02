#!/usr/bin/env python3

import argparse
from igraph import Graph
import random
import pickle
import numpy as np
import os


def generate_dynamic_graphs_er(num_graphs, num_nodes, num_timesteps, num_edges, num_labels):
    """
    Generate a set of dynamic graphs and corresponding labels using the Erdős-Rényi model.

    Parameters
    ----------
    num_graphs : int
        The number of dynamic graphs to generate.
    num_nodes : int
        The number of nodes in each graph.
    num_timesteps : int
        The number of timesteps in each dynamic graph.
    num_edges : int
        The number of edges to be added at each timestep.
    num_labels : int
        The number of unique labels to assign.

    Returns
    -------
    dynamic_graphs : list of list of Graph
        The generated dynamic graphs. Each dynamic graph is represented as a list of igraph Graph instances.
    labels : list of int
        The labels for the dynamic graphs. A label is 1 if the average edge count in the graph is greater than
        the number of nodes multiplied by the number of edges, and 0 otherwise.
    """
    dynamic_graphs = []
    labels = []
    for _ in range(num_graphs):
        edge_prob = num_edges / (num_nodes * (num_nodes - 1) / 2)  # calculate edge probability
        dynamic_graph = []
        for _ in range(num_timesteps):
            g = Graph.Erdos_Renyi(n=num_nodes, p=edge_prob)
            # Assign node labels
            labels_per_node = [random.randint(1, num_labels) for _ in range(num_nodes)]
            g.vs["label"] = [f"{label}" for i, label in enumerate(labels_per_node)]
            # Assign edge attributes
            g.es["attribute"] = [random.randint(1, 10) for _ in range(g.ecount())]
            dynamic_graph.append(g)
        dynamic_graphs.append(dynamic_graph)
        labels.append(int(g.ecount() > num_nodes * num_edges))
    return dynamic_graphs, labels


def generate_dynamic_graphs_via_random_ba_graphs(num_graphs, num_nodes, num_timesteps, num_edges, num_labels):
    dynamic_graphs = []
    labels = []
    for _ in range(num_graphs):
        dynamic_graph = []
        for _ in range(num_timesteps):
            # Generate graph using Barabási–Albert model
            g = Graph.Barabasi(n=num_nodes, m=num_edges)

            # Assign node labels
            labels_per_node = [random.randint(1, num_labels) for _ in range(num_nodes)]
            g.vs["label"] = [f"{label}" for i, label in enumerate(labels_per_node)]

            # Assign edge attributes
            g.es["attribute"] = [random.randint(1, 10) for _ in range(g.ecount())]

            dynamic_graph.append(g)

        dynamic_graphs.append(dynamic_graph)

        # Generate label based on the total number of edges in the graph
        labels.append(int(g.ecount() > num_nodes * num_edges))

    return dynamic_graphs, labels

def generate_dynamic_graphs_via_ba_growth(num_graphs, initial_nodes, num_timesteps, num_edges, num_labels):
    dynamic_graphs = []
    labels = []
    for _ in range(num_graphs):
        dynamic_graph = []

        # Start with an initial complete graph
        g = Graph.Full(initial_nodes)
        g.vs["label"] = [str(random.randint(1, num_labels)) for _ in range(initial_nodes)]
        g.es["attribute"] = [random.randint(1, 10) for _ in range(g.ecount())]

        dynamic_graph.append(g.copy())

        for _ in range(num_timesteps):
            # Add a new node
            g.add_vertices(1)

            # Connect the new node to existing nodes based on the preferential attachment rule
            degrees = g.degree()
            probabilities = [deg / sum(degrees) for deg in degrees]
            targets = np.random.choice(range(len(degrees)), size=num_edges, p=probabilities, replace=False)

            for t in targets:
                g.add_edges([(len(degrees)-1, t)])

            # Assign node label for the new node
            g.vs[len(degrees)-1]["label"] = str(random.randint(1, num_labels))

            # Assign edge attributes for the new edges
            g.es[g.ecount()-num_edges:]["attribute"] = [random.randint(1, 10) for _ in range(num_edges)]

            dynamic_graph.append(g.copy())

        dynamic_graphs.append(dynamic_graph)

        # Generate label based on the total number of edges in the graph
        labels.append(int(g.ecount() > initial_nodes * num_edges))

    return dynamic_graphs, labels


def save_dynamic_graphs(dynamic_graphs, labels, path):
    """
    Save each dynamic graph and its label into a separate pickle file.

    Parameters
    ----------
    dynamic_graphs : list of list of Graph
        The generated dynamic graphs.
    labels : list of int
        The labels for the dynamic graphs.
    path : str
        The directory path where the pickle files will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for i, (graph, label) in enumerate(zip(dynamic_graphs, labels)):
        filename = os.path.join(path, f"dynamic_graph_{i}.pkl")
        with open(filename, "wb") as f:
            pickle.dump((graph, label), f)


def main():
    parser = argparse.ArgumentParser(description='Generate and save dynamic graphs.')
    parser.add_argument('--type', type=str, default="barabasi_albert_growth", help='Type of dynamic graph to generate (erdos_renyi, barabasi_albert_random, baraba_albert_growth)')
    parser.add_argument('--num-graphs', type=int, default=10, help='Number of graphs to generate')
    parser.add_argument('--initial-nodes', type=int, default=5, help='Number of initial nodes in the graph')
    parser.add_argument('--num-timesteps', type=int, default=10, help='Number of timesteps')
    parser.add_argument('--num-edges', type=int, default=2, help='Number of edges to add at each timestep')
    parser.add_argument('--num-labels', type=int, default=2, help='Number of possible labels for nodes')

    args = parser.parse_args()

    # Generate the dynamic graphs
    method_mapping = {
        "erdos_renyi": generate_dynamic_graphs_er,
        "barabasi_albert_random": generate_dynamic_graphs_via_random_ba_graphs,
        "barabasi_albert_growth": generate_dynamic_graphs_via_ba_growth
    }
    selected_method = method_mapping[args.type]
    dynamic_graphs, labels = selected_method(
        args.num_graphs, 
        args.initial_nodes, 
        args.num_timesteps, 
        args.num_edges, 
        args.num_labels
    )

    # Save the dynamic graphs
    save_dynamic_graphs(dynamic_graphs, labels, f"./data/labeled_datasets/{args.type}/")

if __name__ == "__main__":
    main()