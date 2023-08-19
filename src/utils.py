# adapted from https://github.com/BorgwardtLab/filtration_curves
from sklearn.metrics import accuracy_score
import numpy as np


def create_metric_dict(metrics=["accuracy", "training_time", "inference_time"]):
    """
    Creates a dictionary that will store the metrics calculated in cross
    validation.

    Parameters
    ----------
    metrics: list
        Metrics that will be used to assess performance of the
        classifier.

    Returns
    -------
    metric_dict : dict
        Empty dictionary with the metrics of interest as keys.

    """
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = []

    return metric_dict


def compute_fold_metrics(y_test, y_pred, training_time, inference_time, metric_dict):
    """
    Calculates the metrics of interest on the classifier and updates the
    dictionary with the values.

    Given the true values (y_test) and the predicted values of
    a classifier (y_pred), this function computes the metrics of
    interest and updates the current dictionary (metrics_dict) with the value on
    the given fold.

    Parameters
    ----------
    y_test: array-like
        True values of the test data
    y_pred: array-like
        Predicted values from the classifier
    metric_dict: dict
        Dictionary containing the metric of interest and the values so
        far computed on previous folds

    Returns
    -------
    metric_dict : dict
        Updated dictionary values containing the metric of interest and
        its value computed on the current fold

    """
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # update dictionary values
    metric_dict["accuracy"].append(accuracy)
    metric_dict["training_time"].append(training_time)
    metric_dict["inference_time"].append(inference_time)

    return metric_dict


def print_iteration_metrics(iteration_metrics, f=None):
    """
    Prints the mean and standard deviation of the metrics over all
    iterations.

    Parameters
    ----------
    iteration_metrics: dict
        Dictionary of metrics and the mean accuracy on each iteration.
    f: str
        File name to save the results, if desired.

    Returns
    -------

    """
    for metric in iteration_metrics:  #
        if metric in ["training_time", "inference_time"]:
            mean = np.mean(iteration_metrics[metric])
            sdev = np.std(iteration_metrics[metric])
        else:
            mean = np.mean(iteration_metrics[metric]) * 100
            sdev = np.std(iteration_metrics[metric]) * 100
        if f is None:
            print(f"{metric}: {mean:2.4f} +- {sdev:2.4f}")
        else:
            print(f"{metric}: {mean:2.4f} +- {sdev:2.4f}", file=f)

def update_iteration_metrics(fold_metrics, iteration_metrics):
    """
    Updates the dictionary of iteration metrics with the average of the
    fold metrics.

    Updates the list of iteration-level metric results of the
    classifier by appending the mean of the fold metrics.  This is
    necessary when running multiple iterations of k-fold cross
    validation.

    Parameters
    ----------
    fold_metrics: dict
        A dictionary containing the metrics and their results evaluated
        on the individual folds of cross validation.
    iteration_metrics: dict
        A dictionary containing the metrics and the mean results from
        all folds of k-fold cross validation.

    Returns
    -------
    iteration_metrics: dict
        An updated dictionary of the iteration-level metrics, including
        the current iteration.

    """
    for metric in fold_metrics:
        iteration_metrics[metric].append(np.mean(fold_metrics[metric]))

    return iteration_metrics
