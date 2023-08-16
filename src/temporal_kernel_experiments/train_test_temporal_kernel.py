import time
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
    DS_prefix = DS_prefix.split("/")[-1]
    gram_matrix = load_gram_matrix(f"./{DS_prefix}__SEWL_{k_value}.gram")
    return gram_matrix

# 3. Train and test SVM
def train_and_test_svm(DS_prefix, tgkernel_path, labels):
    C_values = [10**i for i in range(-3, 4)]
    k_values = list(range(6))
    
    nested_scores = []

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_cv.split(gram_matrix, labels):
        best_score = -float('inf')
        best_k = -1
        best_c = -1

        # Inner cross-validation for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for k in k_values:
            gram_matrix = compute_gram_matrix(DS_prefix, k, tgkernel_path)
            svm = SVC(kernel='precomputed')
            clf = GridSearchCV(estimator=svm, param_grid={'C': C_values}, cv=inner_cv)
            clf.fit(gram_matrix[train_idx][:, train_idx], labels[train_idx])
            if clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_k = k
                best_c = clf.best_params_['C']

        # Now, use the best_k and best_c to train and assess on the outer split
        gram_matrix = compute_gram_matrix(DS_prefix, best_k, tgkernel_path)
        svm = SVC(kernel='precomputed', C=best_c)
        svm.fit(gram_matrix[train_idx][:, train_idx], labels[train_idx])
        
        score = svm.score(gram_matrix[test_idx][:, train_idx], labels[test_idx])
        nested_scores.append(score)
        
    return np.mean(nested_scores), np.std(nested_scores)

# 4. Test the training and inference speed (without hyperparameter tuning)
def train_and_inference_speed(DS_prefix, tgkernel_path, labels):
    train_times = []
    inference_times = []

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # using same random_state for consistency

    start_gram_time = time.time() # check whether call to tgkernel is synchronous (i.e. time is captured)
    gram_matrix = compute_gram_matrix(DS_prefix, 0, tgkernel_path) # use some fixed k
    gram_matrix_size = gram_matrix.size * 8 / (1024 * 1024) # in MB
    end_gram_time = time.time()
    gram_time = end_gram_time - start_gram_time

    for train_idx, test_idx in outer_cv.split(gram_matrix, labels):
        start_train_time = time.time() # check whether call to tgkernel is synchronous (i.e. time is captured)

        svm = SVC(C=1, kernel='precomputed')  # Use some default value of C
        svm.fit(gram_matrix[train_idx][:, train_idx], labels[train_idx])
        
        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        start_inference_time = time.time()
        _ = svm.score(gram_matrix[test_idx][:, train_idx], labels[test_idx])
        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time

        train_times.append(training_time)
        inference_times.append(inference_time)

    return np.mean(train_times), np.mean(inference_times), gram_time, gram_matrix_size

def compute_accuracies(DS_prefix, tgkernel_path):
    labels = load_data(DS_prefix)
    best_k = -1
    best_accuracy = -float('inf')

    for k in range(6):  # Since k is between 0 and 5
        mean_accuracy, std_dev = train_and_test_svm(DS_prefix, tgkernel_path, labels)
        
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k
            
        print(f"Mean Accuracy for k={k}: {mean_accuracy}, Std Dev: {std_dev}")

    print(f"Best k-value: {best_k} with accuracy: {best_accuracy}")

def compute_times(DS_prefix, tgkernel_path):
    labels = load_data(DS_prefix)
    mean_train_time, mean_inference_time, gram_time, gram_matrix_size = train_and_inference_speed(DS_prefix, tgkernel_path, labels)
    print(f"Gram Matrix Time: {gram_time:.4f}s")
    print(f"Mean Training Time: {mean_train_time:.4f}s")
    print(f"Mean Inference Time: {mean_inference_time:.4f}s")
    print(f"Gram Matrix Size: {gram_matrix_size:.4f}MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test temporal graph kernels.')
    parser.add_argument('--path', type=str, help='Path to your dataset including the prefix')
    parser.add_argument('--tgkernel', type=str, help='Path to tgkernel')
    parser.add_argument('--metric', type=str, default="accuracy", help='Metric to compute (accuracy, time)')
    args = parser.parse_args()
    if (args.metric == "accuracy"):
        compute_accuracies(args.path, args.tgkernel)
    elif (args.metric == "time"):
        compute_times(args.path, args.tgkernel)
    else:
        print(f"Unsupported metric: {args.metric}")