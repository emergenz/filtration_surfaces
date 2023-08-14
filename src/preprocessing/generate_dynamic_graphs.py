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
        dynamic_graph = []
        for i in range(num_timesteps):
            edge_prob = (num_edges + 2*i) / (num_nodes * (num_nodes - 1) / 2)  # calculate edge probability
            edge_prob = min(edge_prob, 1.0)
            g = Graph.Erdos_Renyi(n=num_nodes, p=edge_prob)
            # Assign node labels
            labels_per_node = [random.randint(1, num_labels) for _ in range(num_nodes)]
            g.vs["label"] = [f"{label}" for i, label in enumerate(labels_per_node)]
            # Assign edge attributes
            g.es["weight"] = [random.randint(1, 10) for _ in range(g.ecount())]
            dynamic_graph.append(g)
        dynamic_graphs.append(dynamic_graph)
        print(int(g.vs["label"].count('1') > g.vcount() / 2))
        labels.append(int(g.vs["label"].count('1') > g.vcount() / 2))
    return dynamic_graphs, labels


def generate_dynamic_graphs_via_random_ba_graphs(num_graphs, num_nodes, num_timesteps, num_edges, num_labels):
    dynamic_graphs = []
    labels = []
    for _ in range(num_graphs):
        dynamic_graph = []
        for i in range(num_timesteps):
            # Generate graph using Barabási–Albert model
            g = Graph.Barabasi(n=num_nodes, m=(num_edges + 2 * i))

            # Assign node labels
            labels_per_node = [random.randint(1, num_labels) for _ in range(num_nodes)]
            g.vs["label"] = [f"{label}" for i, label in enumerate(labels_per_node)]

            # Assign edge attributes
            g.es["weight"] = [random.randint(1, 10) for _ in range(g.ecount())]

            dynamic_graph.append(g)

        dynamic_graphs.append(dynamic_graph)

        print(int(g.vs["label"].count('1') > g.vcount() / 2))
        labels.append(int(g.vs["label"].count('1') > g.vcount() / 2))

    return dynamic_graphs, labels

def generate_dynamic_graphs_via_ba_growth(num_graphs, initial_nodes, num_timesteps, num_edges, num_labels):
    dynamic_graphs = []
    labels = []
    for _ in range(num_graphs):
        dynamic_graph = []

        # Start with an initial complete graph
        g = Graph.Full(initial_nodes)
        g.vs["label"] = [str(random.randint(1, num_labels)) for _ in range(initial_nodes)]
        g.es["weight"] = [random.randint(1, 10) for _ in range(g.ecount())]

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
            g.es[g.ecount()-num_edges:]["weight"] = [random.randint(1, 10) for _ in range(num_edges)]

            dynamic_graph.append(g.copy())

        dynamic_graphs.append(dynamic_graph)

        print(int(np.mean(g.es["weight"]) > 5))
        labels.append(int(np.mean(g.es["weight"]) > 5))
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

def save_dynamic_graphs_txt_format(dynamic_graphs, labels, path, dataset_name="DS"):
    """
    Save each dynamic graph and its label into the specified format.
    """
    os.makedirs(path, exist_ok=True)
    
    prefix = os.path.join(path, dataset_name)

    # File paths
    a_path = f"{prefix}_A.txt"
    graph_indicator_path = f"{prefix}_graph_indicator.txt"
    graph_labels_path = f"{prefix}_graph_labels.txt"
    node_labels_path = f"{prefix}_node_labels.txt"
    edge_attributes_path = f"{prefix}_edge_attributes.txt"
    info_path = f"{prefix}_info.txt"

    # Dictionary to track first appearance of each edge
    edge_to_time = {}

    # Counters
    total_nodes = 0
    total_graphs = len(dynamic_graphs)

    # Go through graphs and timesteps
    for graph_id, graph_list in enumerate(dynamic_graphs, 1):
        for timestep, graph in enumerate(graph_list):
            for edge in graph.es:
                edge_tuple = (edge.source + total_nodes + 1, edge.target + total_nodes + 1)
                if edge_tuple not in edge_to_time:
                    edge_to_time[edge_tuple] = timestep
            total_nodes += len(graph.vs)

    # Now, write edges and their first appearance times
    with open(a_path, 'w') as a_file, open(edge_attributes_path, 'w') as ea_file:
        for edge, timestep in edge_to_time.items():
            a_file.write(f"{edge[0]},{edge[1]}\n")
            ea_file.write(f"{timestep}\n")

    # Write nodes, graph indicators and graph labels
    with open(graph_indicator_path, 'w') as gi_file, open(node_labels_path, 'w') as nl_file, open(graph_labels_path, 'w') as gl_file:
        node_counter = 0
        for graph_id, graph_list in enumerate(dynamic_graphs, 1):
            for timestep, graph in enumerate(graph_list):
                for v in graph.vs:
                    node_counter += 1
                    gi_file.write(f"{graph_id}\n")
                    nl_file.write(f"{timestep},{v['label']}\n")
            gl_file.write(f"{labels[graph_id-1]}\n")

    # Write the info file
    total_edges = len(edge_to_time)
    with open(info_path, 'w') as info_file:
        info_file.write(f"Total number of nodes: {total_nodes}\n")
        info_file.write(f"Total number of edges: {total_edges}\n")
        info_file.write(f"Total number of graphs: {total_graphs}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate and save dynamic graphs.')
    parser.add_argument('--type', type=str, default="barabasi_albert_growth", help='Type of dynamic graph to generate (erdos_renyi, barabasi_albert_random, baraba_albert_growth)')
    parser.add_argument('--num-graphs', type=int, default=10, help='Number of dynamic graphs to generate')
    parser.add_argument('--initial-nodes', type=int, default=5, help='Number of initial nodes in the graph')
    parser.add_argument('--num-timesteps', type=int, default=10, help='Number of timesteps')
    parser.add_argument('--num-edges', type=int, default=2, help='Number of edges to add at each timestep')
    parser.add_argument('--num-labels', type=int, default=2, help='Number of possible labels for nodes')
    parser.add_argument('--format', type=str, default="pickle", help='Format of the saved dynamic graphs (pickle, txt)')

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
    if args.format == "pickle":
        save_dynamic_graphs(dynamic_graphs, labels, f"./data/labeled_datasets/{args.type}/")
    elif args.format == "txt":
        save_dynamic_graphs_txt_format(dynamic_graphs, labels, f"./data/labeled_datasets/{args.type}/")
    else:
        print(f"Unsupported format: {args.format}")

if __name__ == "__main__":
    main()