# adapted from https://github.com/BorgwardtLab/filtration_curves
import time
import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from relabel_edges import relabel_edges_with_curvature

def filtration_by_edge_attribute(
    graph, attribute="weight", delete_nodes=True, stop_early=False
):
    """
    Calculates a filtration of a graph based on an edge attribute of the
    graph.

    :param graph: Graph
    :param attribute: Edge attribute name
    :param delete_nodes: If set, removes nodes from the filtration if
    none of their incident edges is part of the subgraph. By default,
    all nodes are kept.
    :param stop_early: If set, stops the filtration as soon as the
    number of nodes has been reached.

    :return: Filtration as a list of tuples, where each tuple consists
    of the weight threshold and the graph.
    """

    weights = graph.es[attribute]
    weights = np.array(weights)

    if len(weights.shape) == 2 and weights.shape[1] == 1:
        weights = weights.squeeze()
    elif len(weights.shape) != 1:
        raise RuntimeError("Unexpected edge attribute shape")

    # Represents the filtration of graphs according to the
    # client-specified attribute.
    F = []

    n_nodes = graph.vcount()

    if (
        weights.size != 1
    ):  # hack to deal with funny graph that has a single edge and was getting 0-D errors
        weights = weights
        x = False
    else:
        weights = np.array([[weights]])
        x = True
        # weights = [weights]

    for weight in sorted(weights):
        if x:  # again part of the hack
            weight = weight[0]
        edges = graph.es.select(lambda edge: edge[attribute] <= weight)
        subgraph = edges.subgraph(delete_vertices=delete_nodes)

        # Store weight and the subgraph induced by the selected edges as
        # one part of the filtration. The client can decide whether each
        # node that is not adjacent to any edge should be removed or not
        # in the filtration (see above).
        F.append((weight, subgraph))

        # If the graph has been filled to the brim with nodes already,
        # there is no need to continue.
        if stop_early and subgraph.vcount() == n_nodes:
            break

    return F

def node_label_distribution(filtration, label_to_index):
    """
    Calculates the node label distribution along a filtration.

    Given a filtration from an individual graph, we calculate the node
    label histogram (i.e. the count of each unique label) at each step
    along that filtration, and returns a list of the weight of the filtration and
    its associated count vector.

    Parameters
    ----------
    filtration : list
        A filtration of graphs
    label_to_index : mappable
        A map between labels and indices, required to calculate the
        histogram.

    Returns
    -------
    D : list
        Label distributions along the filtration. Each entry is a tuple
        consisting of the weight of the filtration followed by a count
        vector.

    """
    # Will contain the distributions as count vectors; this is
    # calculated for every step of the filtration.
    D = []

    for weight, graph in filtration:
        labels = graph.vs["label"]
        # TODO: why an array? we could just directly use a dict (hashmap) mapping from label to count
        counts = np.zeros(len(label_to_index))

        for label in labels:
            index = label_to_index[label]
            counts[index] += 1

        # The conversion ensures that we can serialise everything later
        # on into a `pd.series`.
        D.append((weight, counts.tolist()))

    return D

def save_curves_for_dynamic_graphs(
    source_path="./data/labeled_datasets/BZR_MD",
    output_path="./data/preprocessed_data/BZR_MD/",
):
    """
    Creates the node label filtration curves for dynamic graphs.

    Parameters
    ----------
    source_path: str
        The path to the dataset, which are sequences of igraphs
        (representing dynamic graphs) stored in a pickle file.
    output_path: str
        The path to the output directory where the filtrations will be
        saved as a csv (one dynamic graph per directory, with each
        timestamp saved as a csv)

    Returns
    -------

    """
    # get all file names
    filenames = sorted(glob.glob(os.path.join(source_path, "*.pkl")))

    all_node_labels = set()

    for filename in tqdm(filenames):
        # Each pickle file contains a sequence of igraphs
        dynamic_graph, label = pickle.load(open(filename, "rb"))

        # Update all_node_labels with labels from current graph
        for graph in dynamic_graph:
            all_node_labels.update(map(int, graph.vs["label"]))

    # Create sorted list of all possible node labels
    all_node_labels = sorted(all_node_labels)

    surface_creation_time = 0
    for filename in tqdm(filenames):
        # Each pickle file contains a sequence of igraphs
        dynamic_graph, label = pickle.load(open(filename, "rb"))

        # We create a new directory for each dynamic graph
        dynamic_graph_dir = os.path.join(
            output_path, os.path.splitext(os.path.basename(filename))[0]
        )

        # check if weights exist, if not, compute curvature
        if "weight" not in dynamic_graph[0].es.attributes():
                dynamic_graph = relabel_edges_with_curvature(dynamic_graph)

        surface_creation_start = time.time()
        for idx, graph in enumerate(dynamic_graph):
            # Generate the filtration curve for each graph in the dynamic graph
            # We will store this filtration curve as a separate csv for each timestamp

            if "weight" not in graph.es.attributes():
                graph.es["weight"] = [e["attribute"] for e in graph.es]

            # set all graph labels to integers if they are strings
            for v in graph.vs:
                v["label"] = int(v["label"])

            # TODO: why are we sorting again?
            label_to_index = {
                label: index for index, label in enumerate(all_node_labels)
            }

            # build the filtration using the edge weights
            filtrated_graph = filtration_by_edge_attribute(
                graph, attribute="weight", delete_nodes=True, stop_early=True
            )

            distributions = node_label_distribution(filtrated_graph, label_to_index)

            rows = []
            last_row = None
            for weight, counts in distributions:
                row = {"graph_label": label, "weight": weight}

                row.update(
                    {
                        str(node_label): count
                        for node_label, count in zip(all_node_labels, counts)
                    }
                )
                if row != last_row:
                    rows.append(row)
                    last_row = row

            df = pd.DataFrame(rows)

            # Use reindex to ensure all columns are present
            df = df.reindex(columns=['graph_label', 'weight'] + list(map(str, all_node_labels)))
            df = df.fillna(0.0)

            # We name each csv based on its timestamp
            output_name = "{}.csv".format(idx)
            output_name = os.path.join(dynamic_graph_dir, output_name)

            os.makedirs(dynamic_graph_dir, exist_ok=True)

            df.to_csv(output_name, index=False)
        surface_creation_end = time.time()
        surface_creation_time += (surface_creation_end - surface_creation_start)
    return surface_creation_time


def create_surfaces(args):
    """
    Creates the node label histogram filtration surfaces.

    Creates a node label histogram filtration surface, either by loading
    the previously generated filtration surfaces, or by calling
    save_curves_for_dynamic_graphs(), which will generate the surfaces and
    save them as a csv file.

    Parameters
    ----------
    args: dict
        Command line arguments, used to determine the dataset

    Returns
    -------
    surfaces: list
        A list of node label filtration surfaces, each stored as
        a list of pd.DataFrames
    y: list
        List of graph labels, necessary for classification.
    column_names: list
        List of column names (i.e. each unique node label)

    """
    surface_creation_time = -1
    # check if filtration surfaces are already saved. If not, generate
    # them and save them.
    if not os.path.exists("./data/preprocessed_data/" + args.dataset + "/"):
        surface_creation_time = save_curves_for_dynamic_graphs(
            source_path="./data/labeled_datasets/" + args.dataset + "/",
            output_path="./data/preprocessed_data/" + args.dataset + "/",
        )

    # load saved surfaces (faster processing)
    surfaces, y, column_names = load_surfaces(args)

    return surfaces, y, column_names, surface_creation_time


def load_surfaces(args):
    """
    Loads the precomputed node label filtration surfaces from csv files.

    Parameters
    ----------
    args: dict
        Command line arguments, used to determine the dataset

    Returns
    -------
    surfaces: list
        A list of node label filtration surfaces, each stored as
        a list of pd.DataFrame
    y: list
        List of graph labels, necessary for classification.
    column_names: list
        List of column names (i.e. each unique node label)

    """
    # stores the graphs and graph labels
    folders = sorted(
        glob.glob(
            os.path.join(
                "./data/preprocessed_data/" + args.dataset + "/",
                "*",
            )
        )
    )

    y = []
    surfaces = []

    # create list of dataframes (i.e. data) and y labels
    for idx, folder in enumerate(folders):
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))

        surface = []

        # only add y once per dynamic graph
        for i, file in enumerate(files):
            temp_df = pd.read_csv(file, header=0, index_col="weight")
            if not temp_df["graph_label"].empty:  # Add this check
                y.append(temp_df["graph_label"].values[0])
                break
            else:
                print(f"Warning: 'graph_label' column in file {file} is empty.")

        for file in files:
            df = pd.read_csv(file, header=0, index_col="weight")
            df = df.loc[~df.index.duplicated(keep="last")]
            df = df.drop(columns="graph_label")
            surface.append(df)
        surfaces.append(surface)
    y = LabelEncoder().fit_transform(y)

    # get the column names
    example_file = list(pd.read_csv(files[0], header=0, index_col="weight").columns)
    column_names = [c for c in example_file if c not in ["graph_label"]]

    return surfaces, y, column_names


def index_train_data_surfaces(train_files, column_names):
    """
    Creates a common index of weights based on the training data.

    Takes a dataset of filtrations, with each filtration stored as
    a pandas DataFrame, and builds a common index to standardize the
    data. Each filtration gets reindexed with all the thresholds in the
    entire training dataset, and the values of graph descriptor function
    are forward-filled if an individual filtration did not have
    a specific value.

    Parameters
    ----------
    train_files : list of lists
        A list of lists of pd.DataFrames where the edge weight is the index of
        the dataframe, and the columns are the node label histograms
    column_names: list
        A list of column names from the pd.DataFrames in train_files

    Returns
    -------
    X : list of lists
        A list of the reindexed filtrations.
    df_index: index
        The union of all edge weights included in the training data.

    """
    X = [[[] for j in column_names] for i in range(len(train_files))]

    # Create shared index of all training thresholds
    df_index = None
    for dynamic_graph in train_files:
        for df in dynamic_graph:
            if df_index is None:
                df_index = df.index
            else:
                df_index = df.index.union(df_index)
                df_index = df_index[~df_index.duplicated()]

    # reindex with the full training thresholds
    for i, dynamic_graph in enumerate(train_files):
        for df in dynamic_graph:
            tmp_df = df.reindex(df_index)  # create missing values
            tmp_df = tmp_df.fillna(method="ffill")  # forward-filling for consistency
            tmp_df = tmp_df.fillna(0)  # replace values at the beginning

            for j, col in enumerate(column_names):
                X[i][j].append(tmp_df.iloc[:, j].transpose().to_numpy())

    return X, df_index
