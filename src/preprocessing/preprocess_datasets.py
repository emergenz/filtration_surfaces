import argparse
from collections import defaultdict
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx


def convert_dataset_to_dynamic_graph(path, dataset_name="infectious_ct1"):
    # Load the dataset files
    A = pd.read_csv(f"{path}/{dataset_name}_A.txt", header=None).values
    graph_indicator = pd.read_csv(f"{path}/{dataset_name}_graph_indicator.txt", header=None).values.squeeze()
    graph_labels = pd.read_csv(f"{path}/{dataset_name}_graph_labels.txt", header=None).values.squeeze()
    node_labels = pd.read_csv(f"{path}/{dataset_name}_node_labels.txt", header=None, dtype=str)  # load as DataFrame with a single column of strings
    node_labels = node_labels[0].str.split(',', expand=False)  # split each line into a list of strings
    node_labels = node_labels.apply(lambda x: [int(i) for i in x])  # convert strings to integers
    edge_attributes = pd.read_csv(f"{path}/{dataset_name}_edge_attributes.txt", header=None).values.squeeze()
    # Get unique graph ids
    graph_ids = np.unique(graph_indicator)

    dynamic_graphs = defaultdict(list)
    labels = defaultdict(list)

    for graph_id in graph_ids:
        graph_nodes = np.where(graph_indicator == graph_id)[0] + 1  # node ids are 1-indexed
        graph_nodes_set = set(graph_nodes)  # convert to set for faster lookups
        graph_edges = [(A[i, 0], A[i, 1]) for i in range(len(A)) if A[i, 0] in graph_nodes_set and A[i, 1] in graph_nodes_set]
        
        # construct graph timeline
        graph_timeline = defaultdict(lambda: {"nodes": set(), "edges": set()})  # change "edges" to a set
        for node in graph_nodes:
            node_label_data = node_labels[node-1]  # get the list of timestamps and labels for this node
            for i in range(0, len(node_label_data) - 1, 2):  # iterate over pairs, avoid index out of range
                timestamp, label = node_label_data[i], node_label_data[i+1]
                graph_timeline[timestamp]["nodes"].add((node, label))
                
        for i, edge in enumerate(graph_edges):
            timestamp = edge_attributes[i]
            graph_timeline[timestamp]["edges"].add(edge)

        # construct networkx graphs for each timestamp, keep edges from previous timestamps
        current_edges = set()
        for timestamp, graph_data in sorted(graph_timeline.items()):
            current_edges.update(graph_data["edges"])  # add new edges to current_edges
            g = nx.Graph()
            g.add_nodes_from([node for node in graph_data["nodes"]])
            g.add_edges_from(current_edges)  # use current_edges instead of graph_data["edges"]
            nx.set_node_attributes(g, {node: {'label': label} for node, label in graph_data["nodes"]})
            dynamic_graphs[graph_id].append(g)
            labels[graph_id].append(graph_labels[graph_id-1])  # graph ids are 1-indexed

    return dynamic_graphs, labels

# TODO: this is replicated here and in generate_dynamic_graphs.py
# Save the dynamic graphs and their labels like in save_dynamic_graphs()
def save_dynamic_graphs(dynamic_graphs, labels, path):
    """
    Save dynamic graphs and their labels to a single pickle file.

    Parameters:
    graphs (list): A list of networkx graphs representing the dynamic graphs.
    labels (list): A list of labels for the dynamic graphs.
    filename (str): The name of the pickle file to save the dynamic graphs and labels to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for i, (graph, label) in enumerate(zip(dynamic_graphs, labels)):
        filename = os.path.join(path, f"dynamic_graph_{i}.pkl")
        with open(filename, "wb") as f:
            pickle.dump((graph, label), f)

def main():
    parser = argparse.ArgumentParser(description='Convert and save dynamic graphs.')
    parser.add_argument('--dataset', type=str, default="infectious_ct1", help='Name of the dataset to convert')
    parser.add_argument('--path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    # Generate the dynamic graphs
    dynamic_graphs, labels = convert_dataset_to_dynamic_graph(
        args.path, args.dataset
    )

    # Save the dynamic graphs
    save_dynamic_graphs(dynamic_graphs, labels, f"./data/labeled_datasets/{args.dataset}/")

if __name__ == "__main__":
    main()