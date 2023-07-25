import argparse
import random
import os
import glob
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from utils import create_metric_dict, compute_fold_metrics, print_iteration_metrics, update_iteration_metrics
from graph_processing import filtration_by_edge_attribute, node_label_distribution, save_curves_for_dynamic_graphs, create_surfaces, load_surfaces, index_train_data_surfaces

def run_rf(X, y, n_iterations=10):
    random.seed(42)

    iteration_metrics = create_metric_dict()
    iteration_accuracies = []
    for iteration in range(n_iterations):
        fold_metrics = create_metric_dict()
        fold_accuracies = []

        cv = StratifiedKFold(n_splits=10, random_state=42 + iteration, shuffle=True)

        for train_index, test_index in cv.split(np.zeros(len(y)), y):
            X_train = [X[i] for i in train_index]
            y_train = [y[i] for i in train_index]

            X_test = [X[i] for i in test_index]
            y_test = [y[i] for i in test_index]

            clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=1000, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fold_metrics = compute_fold_metrics(y_test, y_pred, fold_metrics)
        iteration_metrics = update_iteration_metrics(fold_metrics, iteration_metrics)

    print_iteration_metrics(iteration_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset")
    parser.add_argument("--method", default="transductive", type=str, help="transductive or inductive")
    args = parser.parse_args()

    # generate the filtration surfaces (saved to csv for easier handling)
    surfaces, y, column_names = create_surfaces(args)

    n_dynamic_graphs = len(y)
    n_node_labels = surfaces[0][0].shape[1]  # Assume all graphs have the same number of node labels

    if args.method == "transductive":
        surfaces, index = index_train_data_surfaces(surfaces, column_names)
        X = []

        # longest timestamp, so that all input vectors have the same dimension
        for dynamic_graph_idx in range(n_dynamic_graphs):
            dynamic_graph_representation = []
            for node_label_idx in range(n_node_labels):
                # of each timestamp
                for timestamp in range(len(surfaces[dynamic_graph_idx][node_label_idx])):
                    dynamic_graph_representation.extend(surfaces[dynamic_graph_idx][node_label_idx][timestamp])
            X.append(dynamic_graph_representation)

        run_rf(X, y)

    elif args.method == "inductive":
        X = surfaces
        # Replace run_rf_inductive with the actual function
        # run_rf_inductive(X, y, column_names=column_names)
