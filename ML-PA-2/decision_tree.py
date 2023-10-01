# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
import graphviz


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    try:
        unique_values, indices_by_value = np.unique(x), {}
        for value in unique_values:
            indices_by_value[value] = np.where(x == value)[0]
        return indices_by_value
    except Exception as e:
        print("An exception occurred:", e)


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    try:
        probabilities, y_entropy_value = [], 0
        val_indices = partition(y)
        num_samples = len(y)
        for val in val_indices.keys():
            count = len(val_indices[val])
            probabilities.append(float(count/num_samples))
        for prob in probabilities:
            y_entropy_value = y_entropy_value - prob * np.log2(prob)
        return y_entropy_value
    except Exception as e:
        print("An exception occurred:", e)



def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    try:
        y_entropy = entropy(y)
        y_entropy_given_x = 0
        num_samples = len(x)
        val_indices = partition(x)

        for val in val_indices.keys():
            probability = float(len(val_indices[val]) / num_samples)
            y_given_x = [y[i] for i in val_indices[val]]
            y_entropy_given_x += probability * entropy(y_given_x)

        return y_entropy - y_entropy_given_x
    except Exception as e:
        print("An exception occurred:", e)




def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a probabilitylem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    try:
        if attribute_value_pairs is None:
            attribute_value_pairs = []
            for attr_idx in range(len(x[0])):
                all_values = np.array([item[attr_idx] for item in x])
                for attr_value in np.unique(all_values):
                    attribute_value_pairs.append((attr_idx, attr_value))

        attribute_value_pairs = np.array(attribute_value_pairs)

        unique_vals_y = np.unique(y)
        if len(unique_vals_y) == 1:
            return unique_vals_y[0]

        unique_vals_y, count_y = np.unique(y, return_counts=True)
        if len(attribute_value_pairs) == 0:
            return unique_vals_y[np.argmax(count_y)]

        if max_depth == depth:
            return unique_vals_y[np.argmax(count_y)]

        mutual_info_pairs = []
        for attr, val in attribute_value_pairs:
            attr_val_arr = np.array((x[:, attr] == val).astype(int))
            mutual_info = mutual_information(attr_val_arr, y)
            mutual_info_pairs.append(mutual_info)

        mutual_info_pairs = np.array(mutual_info_pairs)
        chosen_attr, chosen_val = attribute_value_pairs[np.argmax(mutual_info_pairs)]

        part = partition(np.array((x[:, chosen_attr] == chosen_val).astype(int)))
        attribute_value_pairs = np.delete(attribute_value_pairs, np.argmax(mutual_info_pairs), 0)

        decision_tree = {}
        for val, ele_indices in part.items():
            out_label = bool(val)
            x_after_part = x.take(np.array(ele_indices), axis=0)
            y_after_part = y.take(np.array(ele_indices), axis=0)
            decision_tree[(chosen_attr, chosen_val, out_label)] = id3(x_after_part, y_after_part,
                                                                      attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)

        return decision_tree
    except:
        raise Exception('Function not yet implemented!')



def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    try:
        for node, child in tree.items():
            feature_idx = node[0]
            feature_val = node[1]
            decision = node[2]

            if decision == (x[feature_idx] == feature_val):
                if type(child) is not dict:
                    predicted_label = child
                else:
                    predicted_label = predict_example(x, child)

        return predicted_label
    except Exception as e:
        print("An exception occurred:", e)



def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    try:
        n = len(y_true)
        return np.sum(np.abs(y_true-y_pred))/n
    except:
        raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def confusion_matrix_multiclass(y, y_pred, classes, fig):
    confusion_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))))
    rows = []
    columns = []
    for cl in classes.tolist():
        rows.append("Actual " + str(cl))
        columns.append("Predicted " + str(cl))
    for i, j in zip(y, y_pred):
        # breakpoint()
        confusion_matrix[i][j] += 1
    fig.subplots_adjust(left=0.3, top=0.8, wspace=2)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=2, rowspan=2)
    table = ax.table(cellText=confusion_matrix.tolist(),
                     rowLabels=rows,
                     colLabels=columns, loc="upper center")
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis("off")


def confusion_matrix(y, y_pred, fig):
    confusion_matrix = np.zeros((2, 2))
    rows = ["Actual Positive", "Actual Negative"]
    cols = ("Classifier Positive", "Classifier Negative")
    for i in range(len(y_pred)):
        if int(y_pred[i]) == 1 and int(y[i]) == 1:
            confusion_matrix[0, 0] += 1 
        elif int(y_pred[i]) == 1 and int(y[i]) == 0:
            confusion_matrix[1, 0] += 1  
        elif int(y_pred[i]) == 0 and int(y[i]) == 1:
            confusion_matrix[0, 1] += 1  
        elif int(y_pred[i]) == 0 and int(y[i]) == 0:
            confusion_matrix[1, 1] += 1  

    fig.subplots_adjust(left=0.3, top=0.8, wspace=1)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=2, rowspan=2)
    ax.table(cellText=confusion_matrix.tolist(),
             rowLabels=rows,
             colLabels=cols, loc="upper center")
    ax.axis("off")


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./data/monks-1.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-1.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    test_error = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(test_error * 100))


    """
    1.) For depth = 1, . . . , 10, learn decision trees and compute the average
    training and test errors on each of the three MONK’s probabilitylems. Make three plots, one for each
    of the MONK’s probabilitylem sets, plotting training and testing error curves together for each probabilitylem,
    with tree depth on the x-axis and error on the y-axis.
    """

    train_files = ['./data/monks-1.train', './data/monks-2.train', './data/monks-3.train']
    test_files = ['./data/monks-1.test', './data/monks-2.test', './data/monks-3.test']

    plt.figure(1, figsize=(16,5)).suptitle("Decision Trees")
    monks_all_trees = []
    for i in range (3):
        # Load the training data
        M = np.genfromtxt(train_files[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(test_files[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        train_errors = []
        test_errors = []
        depths = []
        decision_trees = []
        for j in range (10):
            depth = j + 1
            decision_tree = id3(Xtrn, ytrn, max_depth=depth)
            decision_trees.append(decision_tree)
            YPred_trn = [predict_example(x, decision_tree) for x in Xtrn]
            YPred_tst = [predict_example(x, decision_tree) for x in Xtst]
            train_error = compute_error(ytrn, YPred_trn)
            test_error = compute_error(ytst, YPred_tst)
            depths.append(depth)
            train_errors.append(train_error)
            test_errors.append(test_error)

        monks_all_trees.append(decision_trees)

        splt_idx = 130 + i + 1
        plt.subplot(splt_idx)
        plt.title("Monks " + str(i+1))
        plt.xlabel("Max Depth")
        plt.ylabel("Error")
        plt.grid()
        plt.plot(depths, train_errors, 'o-', color='r', label='Training Examples')
        plt.plot(depths, test_errors, 'o-', color='b', label='Testing Examples')
        plt.legend(loc="best")
        # plt.show()

    """
    2.) For monks-1, report the learned decision tree and the confusion
    matrix on the test set for depth=1 and depth=2. A confusion matrix is a table that is used to
    describe the performance of a classifier on a data set.
    """
    M = np.genfromtxt('./data/monks-1.train', missing_values=0,
                  skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-1.test', missing_values=0,
                    skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    decTreeDepth1 = monks_all_trees[0][0]
    decTreeDepth2 = monks_all_trees[0][1]

    print("___________________________")
    print("Decision Tree for Depth 1 ")
    print("___________________________")

    visualize(decTreeDepth1)
    YpreDep1 = [predict_example(x, decTreeDepth1) for x in Xtst]
    fig1 = plt.figure(2)
    confusion_matrix(ytst, YpreDep1, fig1)
    fig1.suptitle("Decision Tree for Depth=1 Confusion Matrix on Monks 1 Dataset")

    print("___________________________")
    print("Decision Tree for Depth 2 ")
    print("___________________________")

    visualize(decTreeDepth2)
    YpreDep2 = [predict_example(x, decTreeDepth2) for x in Xtst]
    fig2 = plt.figure(3)
    confusion_matrix(ytst, YpreDep2, fig2)
    fig2.suptitle("Decision Tree for Depth=2 Confusion Matrix on Monks 1 Dataset")

    """
    3.) For monks-1, use scikit-learns’s default decision tree algorithm2 to learn
    a decision tree. Visualize the learned decision tree using graphviz3. Report the visualized decision
    tree and the confusion matrix on the test set. Do not change the default parameters.
    """
    scikitDescTree = tree.DecisionTreeClassifier(criterion="entropy")
    scikitDescTree.fit(Xtrn, ytrn)
    predictedYScikit = [scikitDescTree.predict(
        np.array(x).reshape(1, -1))[0] for x in Xtst]

    fig3 = plt.figure(4)
    confusion_matrix(ytst, predictedYScikit, fig3)
    fig3.suptitle(
        "Confusion Matrix for Monks 1 Test Set with Scikit Decision tree")

    src = tree.export_graphviz(scikitDescTree, out_file=None)
    graph = graphviz.Source(src)
    graph.format = 'png'
    graph.render("Monks1")
    print(graph)
    print("______________________________________________________________________")
    

    """
    4.) Repeat steps 2 and 3 with your “own” data set and report the confusion
    matrices. You can use other data sets in the UCI repository. If you encounter continuous features,
    consider a simple discretization strategy to pre-process them into binary features using the mean.
    """
    preprocessing
    file = open('./data/bupa.data') # liver disorder dataset
    data = []


    for line in file:
        features = line.strip().split(',')
        data.append(features)
        
    # Encode categorical features using LabelEncoder
    label_encoder = preprocessing.LabelEncoder()
    X = np.zeros((len(data), 6))
    for i in range(6):
        feat = [item[i] for item in data]
        label_encoder.fit(feat)
        feat_encoded = label_encoder.transform(feat)
        X[:, i] = feat_encoded

    # Encode categorical labels using LabelEncoder and obtain class names
    y = label_encoder.fit_transform([item[6] for item in data])
    classes = label_encoder.classes_

    # Split data into training and testing sets
    train_idx = int(len(X) * 0.8)
    X_train = X[:train_idx].astype(int)
    X_test = X[train_idx:-1].astype(int)
    y_train = y[0:train_idx].astype(int)
    y_test = y[train_idx:-1].astype(int)

    # Build decision tree models using ID3 algorithm and depth 1 and 2
    print("Decision Tree on Liver Disorder Evaluation Dataset for depth = 1")
    tree_depth_1 = id3(X_train, y_train, max_depth=1)
    visualize(tree_depth_1)
    y_pred_depth_1 = [predict_example(x, tree_depth_1) for x in X_test]
    fig_depth_1 = plt.figure(6)
    confusion_matrix_multiclass(y_test, y_pred_depth_1, classes, fig_depth_1)
    fig_depth_1.suptitle("Decision Tree Confusion Matrix on Liver Disorder Evaluation Dataset for depth = 1")

    print("Decision Tree on Liver Disorder Evaluation Dataset for Depth = 2 ")
    tree_depth_2 = id3(X_train, y_train, max_depth=2)
    visualize(tree_depth_2)
    y_pred_depth_2 = [predict_example(x, tree_depth_2) for x in X_test]
    fig_depth_2 = plt.figure(7)
    confusion_matrix_multiclass(y_test, y_pred_depth_2, classes, fig_depth_2)
    fig_depth_2.suptitle("Decision Tree Confusion Matrix on Liver Disorder Evaluation Dataset for Depth = 2 ")

    # Build decision tree model using sklearn's DecisionTreeClassifier
    sci_descision_tree = tree.DecisionTreeClassifier(criterion="entropy")
    sci_descision_tree.fit(X_train, y_train)
    y_pred_sci = [sci_descision_tree.predict(np.array(x).reshape(1, -1))[0] for x in X_test]
    fig_sci = plt.figure(8)
    confusion_matrix_multiclass(y_test, y_pred_sci, classes, fig_sci)
    fig_sci.suptitle("Sklearn Decision Tree Confusion Matrix on Liver Disorder Evaluation Test Set")

    # Export decision tree visualization as PNG file
    dot_data = tree.export_graphviz(sci_descision_tree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("Liver Disorder Evaluation")

    # Show the confusion matrix plots
    plt.show()

