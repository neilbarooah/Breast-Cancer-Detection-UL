"""
The following code runs a SVM on 2 datasets
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from time import strftime
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from plotting import plot_learning_curve, plot_hidden_layer_performance
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import CrossValidator
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster  import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from numpy import array
import os
import struct
import arff
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore', category=DeprecationWarning)


def get_breast_cancer_data():
    test_file = pd.read_csv("wdbc_test.csv")
    train_file = pd.read_csv("wdbc_train.csv")
    train_labels = train_file["Diagnosis"]
    test_labels = test_file["Diagnosis"]
    train_features = train_file.drop("Diagnosis", 1)
    test_features = test_file.drop("Diagnosis", 1)
    return train_features.as_matrix(), train_labels.as_matrix(), test_features.as_matrix(), test_labels.as_matrix()

def get_breast_cancer_data_info_gain():
    test_file = pd.read_csv("wdbc_test.csv")
    train_file = pd.read_csv("wdbc_train.csv")
    train_labels = train_file["Diagnosis"]
    test_labels = test_file["Diagnosis"]
    tr_f = train_file.drop("Diagnosis", 1)
    te_f = test_file.drop("Diagnosis", 1)
    
    train_features = tr_f[["'Worst Perimeter'","'Worst Area'","'Worst Radius'","'Worst Concave Points'","'Mean Concave Points'","'Mean Perimeter'","'Mean Area'","'Mean Radius'","'Mean Concavity'","'SE Area'","'Worst Concavity'",
    "'SE Radius'","'SE Perimeter'","'Worst Compactness'","'Mean Compactness'","'SE Concavity'","'SE Concave Points'","'Worst Texture'","'Mean Texture'","'Worst Symmetry'",
    "'SE Compactness'","'Worst Smoothness'","'Mean Symmetry'","'Mean Smoothness'","'Worst Fractal Dimension'","'SE Fractal Dimension'","'SE Symmetry'","'Mean Fractal Dimension'","'SE Texture'","'SE Smoothness'"]]
    test_features = te_f[["'Worst Perimeter'","'Worst Area'","'Worst Radius'","'Worst Concave Points'","'Mean Concave Points'","'Mean Perimeter'","'Mean Area'","'Mean Radius'","'Mean Concavity'","'SE Area'","'Worst Concavity'",
    "'SE Radius'","'SE Perimeter'","'Worst Compactness'","'Mean Compactness'","'SE Concavity'","'SE Concave Points'","'Worst Texture'","'Mean Texture'","'Worst Symmetry'",
    "'SE Compactness'","'Worst Smoothness'","'Mean Symmetry'","'Mean Smoothness'","'Worst Fractal Dimension'","'SE Fractal Dimension'","'SE Symmetry'","'Mean Fractal Dimension'","'SE Texture'","'SE Smoothness'"]]
   
    return train_features, train_labels, test_features, test_labels

def ann_breast_cancer_pca():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data()
    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_pca.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset (No Dimensionality Reduction)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)

    for n in range(1, 8, 1):    
        pca = PCA(n_components = n)
        pca.fit(tr_x)
        train_features = pca.transform(tr_x)
        test_features = pca.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_pca.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - PCA (" + str(n) + " principal components)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "PCA - 1", "PCA - 2", "PCA - 3", "PCA - 4", "PCA - 5", "PCA - 6", "PCA - 7"])
    plt.title("WDBC: NN Test Accuracy over Epochs with Optimal Hidden Layers")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_pca.png")
    plt.show()


def ann_breast_cancer_ica():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data()
    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_ica.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset (No Dimensionality Reduction)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)

    for n in range(1, 30, 1):    
        ica = FastICA(n_components = n)
        ica.fit(tr_x)
        train_features = ica.transform(tr_x)
        test_features = ica.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_ica.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - ICA (" + str(n) + " independent components)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "ICA - 1", "ICA - 2", "ICA - 3", "ICA - 4", "ICA - 5", "ICA - 6", "ICA - 7",
                "ICA - 8", "ICA - 9", "ICA - 10", "ICA - 11", "ICA - 12", "ICA - 13", "ICA - 14",
                "ICA - 15", "ICA - 16", "ICA - 17", "ICA - 18", "ICA - 19", "ICA - 20", "ICA - 21"
                "ICA - 22", "ICA - 23", "ICA - 24", "ICA - 25", "ICA - 26", "ICA - 27", "ICA - 28"
                "ICA - 29", "ICA - 30"])
    plt.title("WDBC: NN Test Accuracy over Epochs with Optimal Hidden Layers")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_ica.png")
    plt.show()


def ann_breast_cancer_rp():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data()
    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_rp.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset (No Dimensionality Reduction)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)

    for n in range(1, 30, 1):    
        rp = GaussianRandomProjection(n_components = n)
        rp.fit(tr_x)
        train_features = rp.transform(tr_x)
        test_features = rp.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_rp.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - RP (" + str(n) + " random projections)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "RP - 1", "RP - 2", "RP - 3", "RP - 4", "RP - 5", "RP - 6", "RP - 7",
                "RP - 8", "RP - 9", "RP - 10", "RP - 11", "RP - 12", "RP - 13", "RP - 14",
                "RP - 15", "RP - 16", "RP - 17", "RP - 18", "RP - 19", "RP - 20", "RP - 21"
                "RP - 22", "RP - 23", "RP - 24", "RP - 25", "RP - 26", "RP - 27", "RP - 28"
                "RP - 29", "RP - 30"])
    plt.title("WDBC: NN Test Accuracy over Epochs with Optimal Hidden Layers")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_rp.png")
    plt.show()


def ann_breast_cancer_ig():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data()
    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_ig.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset - (No Dimensionality Reduction)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)

    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()

    for n in range(1, 30, 1): 
        cols = tr_x.columns[0:n]
        print(cols)
        train_features = tr_x[cols].as_matrix()
        test_features = te_x[cols].as_matrix()
        train_lables = tr_y.as_matrix()
        test_labels = te_y.as_matrix()

        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_ig.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - IG (top " + str(n) + " features)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "IG - 1", "IG - 2", "IG - 3", "IG - 4", "IG - 5", "IG - 6", "IG - 7",
                "IG - 8", "IG - 9", "IG - 10", "IG - 11", "IG - 12", "IG - 13", "IG - 14",
                "IG - 15", "IG - 16", "IG - 17", "IG - 18", "IG - 19", "IG - 20", "IG - 21"
                "IG - 22", "IG - 23", "IG - 24", "IG - 25", "IG - 26", "IG - 27", "IG - 28"
                "IG - 29", "IG - 30"])
    plt.title("WDBC: NN Test Accuracy over Epochs with Optimal Hidden Layers")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_ig.png")
    plt.show()


def ann_breast_cancer_dr():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data()
    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_summary.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset (No Dimensionality Reduction)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)

    # PCA
    pca_n = [3, 5]
    for n in pca_n:    
        pca = PCA(n_components = n)
        pca.fit(tr_x)
        train_features = pca.transform(tr_x)
        test_features = pca.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_summary.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - PCA (" + str(n) + " principal components)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)


    #ICA
    ica_n = [15, 17]
    for n in ica_n:    
        ica = FastICA(n_components = n)
        ica.fit(tr_x)
        train_features = ica.transform(tr_x)
        test_features = ica.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_summary.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - ICA (" + str(n) + " independent components)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    
    rp_n = [12, 23]
    for n in rp_n:    
        rp = GaussianRandomProjection(n_components = n)
        rp.fit(tr_x)
        train_features = rp.transform(tr_x)
        test_features = rp.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_summary.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - RP (" + str(n) + " random projections)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    # IG
    ig_n = [2, 4]
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()
    for n in ig_n: 
        cols = tr_x.columns[0:n]
        print(cols)
        train_features = tr_x[cols].as_matrix()
        test_features = te_x[cols].as_matrix()
        train_lables = tr_y.as_matrix()
        test_labels = te_y.as_matrix()

        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_summary.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - IG (top " + str(n) + " features)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "PCA (n = 3)", "PCA (n = 5)", "ICA (n = 15)", "ICA (n = 17)", "RP (n = 12)", "RP (n = 23)", "IG (n = 2)", "IG (n = 4)"])
    plt.title("WDBC: NN with DR Test Accuracy")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_summary.png")
    plt.show()


def nn_km():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()

    tr_x = tr_x.as_matrix()
    tr_y = tr_y.as_matrix()
    te_x = te_x.as_matrix()
    te_y = te_y.as_matrix()

    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_km.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset (No Clustering/DR)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)

    
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()
    df = pd.concat([tr_x, te_x])
    kmeans = KMeans(n_clusters=2).fit(df)
    clusters = pd.Series(kmeans.labels_)
    tr_x['Cluster'] = clusters.values[0:427]
    te_x['Cluster'] = clusters.values[427:]

    tr_x = tr_x.as_matrix()
    tr_y = tr_y.as_matrix()
    te_x = te_x.as_matrix()
    te_y = te_y.as_matrix()
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y

    # PCA
    pca_n = [3, 5]
    for n in pca_n:    
        pca = PCA(n_components = n)
        pca.fit(tr_x)
        train_features = pca.transform(tr_x)
        test_features = pca.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_km.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - PCA/KM (" + str(n) + " principal components, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)


    #ICA
    ica_n = [15, 17]
    for n in ica_n:    
        ica = FastICA(n_components = n)
        ica.fit(tr_x)
        train_features = ica.transform(tr_x)
        test_features = ica.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_km.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - ICA/KM (" + str(n) + " independent components, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    
    rp_n = [12, 23]
    for n in rp_n:    
        rp = GaussianRandomProjection(n_components = n)
        rp.fit(tr_x)
        train_features = rp.transform(tr_x)
        test_features = rp.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_km.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - RP/KM (" + str(n) + " random projections, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    # IG
    ig_n = [2, 4]
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()
    for n in ig_n: 
        cols = tr_x.columns[0:n]
        print(cols)
        train_features = tr_x[cols].as_matrix()
        test_features = te_x[cols].as_matrix()
        train_lables = tr_y.as_matrix()
        test_labels = te_y.as_matrix()

        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_km.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - IG/KM (top " + str(n) + " features, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "PCA (n = 3)", "PCA (n = 5)", "ICA (n = 15)", "ICA (n = 17)", "RP (n = 12)", "RP (n = 23)", "IG (n = 2)", "IG (n = 4)"])
    plt.title("WDBC: NN with KM + DR Test Accuracy")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_em_km.png")
    plt.show()



def nn_gmm():
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()

    tr_x = tr_x.as_matrix()
    tr_y = tr_y.as_matrix()
    te_x = te_x.as_matrix()
    te_y = te_y.as_matrix()

    plt.figure()
    n_epochs = 51
    epoch_graph = range(1, 51, 1)
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y
    num_layers = range(10)
    accuracy_train_layer = []
    accuracy_test_layer = []
    start_time = datetime.now()

    for layer in num_layers:
        classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
        classifier.fit(train_features, train_labels)
        accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
        accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

    optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
    classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
        alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

    n_train_samples = train_features.shape[0]
    
    n_batch = 32
    n_classes = np.unique(train_labels)

    scores_train = []
    scores_test = []
    epoch = 1
    while epoch < n_epochs:
        print('epoch: ', epoch)
        random_perm = np.random.permutation(train_features.shape[0])
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        scores_train.append(classifier.score(train_features, train_labels.ravel()))
        scores_test.append(classifier.score(test_features, test_labels.ravel()))

        epoch += 1

    end_time = datetime.now()
    total_time_taken = str(end_time - start_time)
    train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
    cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
    test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

    with open("results/ann_breast_cancer_gmm.txt", 'a') as file:
        file.write("ANN with Breast Cancer Dataset (No Clustering/DR)\n\n")
        file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
        file.write("Training Accuracy: " + str(train_accuracy) + "\n")
        file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
        file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
        file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

    
    plt.plot(epoch_graph, scores_test)


    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()
    df = pd.concat([tr_x, te_x])
    gmm = GaussianMixture(n_components=2).fit(df)
    clusters = gmm.predict(df)
    tr_x['Cluster'] = clusters[0:427]
    te_x['Cluster'] = clusters[427:]

    tr_x = tr_x.as_matrix()
    tr_y = tr_y.as_matrix()
    te_x = te_x.as_matrix()
    te_y = te_y.as_matrix()
    
    # Original
    train_features = tr_x
    train_labels = tr_y
    test_features = te_x
    test_labels = te_y

    # PCA
    pca_n = [3, 5]
    for n in pca_n:    
        pca = PCA(n_components = n)
        pca.fit(tr_x)
        train_features = pca.transform(tr_x)
        test_features = pca.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_gmm.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - PCA/GMM (" + str(n) + " principal components, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)


    #ICA
    ica_n = [15, 17]
    for n in ica_n:    
        ica = FastICA(n_components = n)
        ica.fit(tr_x)
        train_features = ica.transform(tr_x)
        test_features = ica.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_gmm.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - ICA/GMM (" + str(n) + " independent components, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    
    rp_n = [12, 23]
    for n in rp_n:    
        rp = GaussianRandomProjection(n_components = n)
        rp.fit(tr_x)
        train_features = rp.transform(tr_x)
        test_features = rp.transform(te_x)
        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_gmm.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - RP/GMM (" + str(n) + " random projections, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    # IG
    ig_n = [2, 4]
    tr_x, tr_y, te_x, te_y = get_breast_cancer_data_info_gain()
    for n in ig_n: 
        cols = tr_x.columns[0:n]
        print(cols)
        train_features = tr_x[cols].as_matrix()
        test_features = te_x[cols].as_matrix()
        train_lables = tr_y.as_matrix()
        test_labels = te_y.as_matrix()

        num_layers = range(10)
        accuracy_train_layer = []
        accuracy_test_layer = []
        start_time = datetime.now()

        for layer in num_layers:
            classifier = MLPClassifier(hidden_layer_sizes=tuple(layer * [16]), max_iter=600)
            classifier.fit(train_features, train_labels)
            accuracy_train_layer.append(accuracy_score(train_labels, classifier.predict(train_features)))
            accuracy_test_layer.append(accuracy_score(test_labels, classifier.predict(test_features)))

        optimal_num_layers = accuracy_test_layer.index(max(accuracy_test_layer))
        classifier = MLPClassifier(hidden_layer_sizes=(optimal_num_layers * [16]), max_iter=500,
            alpha=1e-4, solver='adam', verbose=0, tol=1e-8, random_state=1)

        n_train_samples = train_features.shape[0]
        
        n_batch = 32
        n_classes = np.unique(train_labels)

        scores_train = []
        scores_test = []
        epoch = 1
        while epoch < n_epochs:
            print('epoch: ', epoch)
            random_perm = np.random.permutation(train_features.shape[0])
            mini_batch_index = 0
            while True:
                indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
                classifier.partial_fit(train_features[indices], train_labels[indices], classes=n_classes)
                mini_batch_index += n_batch

                if mini_batch_index >= n_train_samples:
                    break

            scores_train.append(classifier.score(train_features, train_labels.ravel()))
            scores_test.append(classifier.score(test_features, test_labels.ravel()))

            epoch += 1

        end_time = datetime.now()
        total_time_taken = str(end_time - start_time)
        train_accuracy = accuracy_score(train_labels, classifier.predict(train_features))
        cross_validation_accuracy = cross_val_score(classifier, train_features, train_labels, cv=7).mean()
        test_accuracy = accuracy_score(test_labels, classifier.predict(test_features))

        with open("results/ann_breast_cancer_gmm.txt", 'a') as file:
            file.write("ANN with Breast Cancer Dataset - IG/GMM (top " + str(n) + " features, 2 clusters)\n\n")
            file.write("Optimal Hidden Layers: " + str((optimal_num_layers * [16])) + "\n")
            file.write("Training Accuracy: " + str(train_accuracy) + "\n")
            file.write("Cross Validation Accuracy: " + str(cross_validation_accuracy) + "\n")
            file.write("Testing Accuracy: " + str(test_accuracy) + "\n")
            file.write("Total Time Taken: " + strftime(total_time_taken) + "\n\n\n")

        
        plt.plot(epoch_graph, scores_test)

    plt.legend(["Original", "PCA (n = 3)", "PCA (n = 5)", "ICA (n = 15)", "ICA (n = 17)", "RP (n = 12)", "RP (n = 23)", "IG (n = 2)", "IG (n = 4)"])
    plt.title("WDBC: NN with EM + DR Test Accuracy")
    plt.xlabel("Epochs")
    plt.savefig("nn_breast_cancer_em_dr.png")
    plt.show()


#nn_km()
nn_gmm()