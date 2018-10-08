import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Taken from the following website:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test Accuracy")

    plt.legend(loc="best")
    return plt


def plot_hidden_layer_performance(dataset_name, num_layers, accuracy_train_layer, accuracy_test_layer):
	plt.figure()
	plt.plot(num_layers, accuracy_train_layer, label="Train Accuracy")
	plt.plot(num_layers, accuracy_test_layer, label="Test Accuracy")
	plt.xlabel("No. of Hidden Layers")
	plt.ylabel("Accuracy")
	plt.title("ANN on " + dataset_name + " Dataset")
	plt.legend(loc="best")
	return plt

def plot_knn_k(dataset_name, k, accuracy_train_layer, accuracy_test_layer):
	plt.figure()
	plt.plot(k, accuracy_train_layer, label="Train Accuracy")
	plt.plot(k, accuracy_test_layer, label="Test Accuracy")
	plt.xlabel("K")
	plt.ylabel("Accuracy")
	plt.title("KNN on " + dataset_name + " Dataset")
	plt.legend(loc="best")
	return plt

def plot_dtree_depth_performance(dataset_name, depth, train_accuracy, test_accuracy):
	plt.figure()
	plt.plot(depth, train_accuracy, label="Train Accuracy")
	plt.plot(depth, test_accuracy, label="Test Accuracy")
	plt.xlabel("Max Depth")
	plt.ylabel("Accuracy")
	plt.title("Decision Tree on " + dataset_name + " Dataset")
	plt.legend(loc="best")
	return plt

def plot_boosting_performance(dataset_name, estimators, train_accuracy, test_accuracy):
	plt.figure()
	plt.plot(estimators, train_accuracy, label="Train Accuracy")
	plt.plot(estimators, test_accuracy, label="Test Accuracy")
	plt.xlabel("No. of Estimators")
	plt.ylabel("Accuracy")
	plt.title("Boosting on " + dataset_name + " Dataset")
	plt.legend(loc="best")
	return plt

def plot_svm_performance(dataset_name, kernels, train_accuracy, test_accuracy):
	plt.figure()
	x1 = range(1, 6, 2)
	x2 = range(2, 7, 2)
	plt.style.use('ggplot')
	plt.bar(x1, train_accuracy, color='g', label="Training Accuracy")
	plt.bar(x2, test_accuracy, color='b', label="Testing Accuracy")
	plt.xticks(x1, kernels)
	plt.xticks(x2, kernels)
	plt.xlabel("Kernel")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.title("SVM Accuracy for " + dataset_name + " based on different kernels")
	return plt

