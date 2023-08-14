import os
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from subprocess import run

# 1. Load the dataset
def load_data(DS_prefix):
    graph_labels = np.loadtxt(f"{DS_prefix}_graph_labels.txt", delimiter=",", dtype=int)
    return graph_labels

def load_gram_matrix(filename):
    # Create a list to store the matrix data
    matrix_data = []

    with open(filename, 'r') as f:
        for line in f:
            # Split the line by spaces
            row_data = line.strip().split()
            # Create an empty row for the matrix
            row = np.zeros(len(row_data) - 1)

            for item in row_data[1:]:
                index, value = item.split(':')
                index = int(index) - 1  # 0-based index
                row[index] = float(value)
            matrix_data.append(row)

    # Convert the list of rows to a numpy matrix
    matrix = np.array(matrix_data)
    return matrix

# 2. Compute Gram matrix
def compute_gram_matrix(DS_prefix, k_value, tgkernel_path):
    cmd = f"{tgkernel_path} {DS_prefix} 10 {k_value} {k_value}"
    run(cmd.split())
    gram_matrix = load_gram_matrix(f"./DS__SEWL_{k_value}.gram")
    return gram_matrix

# 3. Train and test SVM
def train_and_test_svm(gram_matrix, labels):
    C_values = [10**i for i in range(-3, 4)]
    param_grid = {'C': C_values}

    svm = SVC(kernel='precomputed')
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Nested CV with parameter optimization
    clf = GridSearchCV(estimator=svm, param_grid=param_grid, cv=inner_cv)
    nested_scores = []

    for train_idx, test_idx in outer_cv.split(gram_matrix, labels):
        clf.fit(gram_matrix[train_idx][:, train_idx], labels[train_idx])
        score = clf.score(gram_matrix[test_idx][:, train_idx], labels[test_idx])
        nested_scores.append(score)
        
    return np.mean(nested_scores), np.std(nested_scores)

def main(DS_prefix, tgkernel_path):
    labels = load_data(DS_prefix)
    best_k = -1
    best_accuracy = -float('inf')

    for k in range(6):  # Since k is between 0 and 5
        gram_matrix = compute_gram_matrix(DS_prefix, k, tgkernel_path)
        mean_accuracy, std_dev = train_and_test_svm(gram_matrix, labels)
        
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k
            
        print(f"Mean Accuracy for k={k}: {mean_accuracy}, Std Dev: {std_dev}")

    print(f"Best k-value: {best_k} with accuracy: {best_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test temporal graph kernels.')
    parser.add_argument('--path', type=str, help='Path to your dataset including the prefix')
    parser.add_argument('--tgkernel', type=str, help='Path to tgkernel')
    args = parser.parse_args()
    main(args.path, args.tgkernel)