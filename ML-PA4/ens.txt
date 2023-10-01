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
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


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


def entropy(y, weights=None):
    """
    Compute the entropy of a vector y by considering the plt_counts of the unique values (v1, ... vk), in z. 
    Include the weights of the boosted examples if present

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    try:
        probabilities, y_entropy_value = [], 0
        val_indices = partition(y)
        num_samples = len(y)
        for val in val_indices.keys():
            if weights is None:
                count = len(val_indices[val])
                probabilities.append(float(count/num_samples))
            else:
                prob = 0
                for idx in val_indices[val]:
                    prob += weights[idx]
                probabilities.append(prob)
        for prob in probabilities:
            y_entropy_value = y_entropy_value - prob * np.log2(prob)
        return y_entropy_value
    except Exception as e:
        print("An exception occurred:", e)


def mutual_information(x, y, weights=None):
    """

    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)

    Compute the weighted mutual information for Boosted learners
    """

    # INSERT YOUR CODE HERE
    try:
        y_entropy = entropy(y, weights)
        y_entropy_given_x = 0
        num_samples = len(x)
        val_indices = partition(x)

        for val in val_indices.keys():
            if weights is None:
                probability = float(len(val_indices[val]) / num_samples)
                weights_new = None
            else:
                probability = 0
                weights_new = []
                for idx in val_indices[val]:
                    probability += weights[idx]
                    weights_new.append(weights[idx])
                probability /= np.sum(weights)
            y_given_x = [y[i] for i in val_indices[val]]
            y_entropy_given_x += probability * entropy(y_given_x, weights_new)

        return y_entropy - y_entropy_given_x
    except Exception as e:
        print("An exception occurred:", e)


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
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
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
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
            mutual_info = mutual_information(attr_val_arr, y, weights)
            mutual_info_pairs.append(mutual_info)

        mutual_info_pairs = np.array(mutual_info_pairs)
        chosen_attr, chosen_val = attribute_value_pairs[np.argmax(
            mutual_info_pairs)]

        part = partition(
            np.array((x[:, chosen_attr] == chosen_val).astype(int)))
        attribute_value_pairs = np.delete(
            attribute_value_pairs, np.argmax(mutual_info_pairs), 0)

        decision_tree = {}
        for val, ele_indices in part.items():
            out_label = bool(val)
            x_after_part = x.take(np.array(ele_indices), axis=0)
            y_after_part = y.take(np.array(ele_indices), axis=0)
            if weights is None:
                decision_tree[(chosen_attr, chosen_val, out_label)] = id3(x_after_part, y_after_part,
                                                                          attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)
            else:
                weights_new = weights.take(np.array(ele_indices), axis=0)
                decision_tree[(chosen_attr, chosen_val, out_label)] = id3(x_after_part, y_after_part,
                                                                          attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth, weights=weights_new)
        return decision_tree
    except Exception as e:
        print("An exception occurred:", e)


def bootstrap_sampler(features, labels, num_samples):
    try:
        sample_indices = np.random.choice(num_samples, num_samples, replace=True)
        sampled_features = features[sample_indices].astype(int)
        sampled_labels = labels[sample_indices].astype(int)
        return sampled_features, sampled_labels
    except Exception as e:
        print(f"An exception occurred: {e}")



def bagging(x, y, max_depth, num_trees):
    """
    Implements bagging of multiple id3 trees where each tree trains on a boostrap sample of the original dataset
    """
    try:
        
        alpha = 1
        h_ens = []
        for tree_index in range(num_trees):
            bootstrap_x, bootstrap_y = bootstrap_sampler(x, y, x.shape[0])
            h_tree = id3(bootstrap_x, bootstrap_y, max_depth=max_depth)
            h_ens.append((alpha, h_tree))
        return h_ens
    except Exception as e:
        print("An exception occurred:", e)


def boosting(x, y, max_depth, num_stumps):
    """
    Implements an AdaBoost algorithm using the ID3 algorithm as a base decision tree
    """
    try:
        ensemble = []  # list of tuples 
        num_samples = x.shape[0]
        sample_weights = np.ones(y.shape) / num_samples  # initialize all weights to 1/N
        for i in range(num_stumps):
            h_l = id3(x, y, max_depth=max_depth, weights=sample_weights)  
            y_pred = [predict_example_3(sample, h_l) for sample in x]  
            eps_t = np.dot(np.absolute(y - y_pred), sample_weights) 
            alpha_t = 0.5 * np.log(((1 - eps_t) / eps_t))  
            indicator = np.absolute(y - y_pred)  
            for idx, w in enumerate(sample_weights):
                if indicator[idx]:
                    sample_weights[idx] *= np.exp(alpha_t)  # increase weight for incorrect predictions
                else:
                    sample_weights[idx] *= np.exp(-alpha_t)  # decrease weight for correct predictions
            sample_weights /= 2 * np.sqrt(eps_t * (1 - eps_t))  
            ensemble.append((alpha_t, h_l))  # add tree and weight to ensemble
        return ensemble
    except Exception as e:
        print("An exception occurred:", e)


def predict_example_3(x, tree):
    
    for decision_node, child_tree in tree.items():
        idx = decision_node[0]
        val = decision_node[1]
        decision = decision_node[2]

        if decision == (x[idx] == val):
            if type(child_tree) is not dict:
                class_label = child_tree
            else:
                class_label = predict_example_3(x, child_tree)

            return class_label


def predict_example_ens(x, h_ens):
    """
    Predicts the classification label for a single example x using a combination of weighted trees
    Returns the predicted label of x according to tree
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    try:
        def predict_example(x, tree):
    
            for decision_node, child_tree in tree.items():
                idx = decision_node[0]
                val = decision_node[1]
                decision = decision_node[2]

                if decision == (x[idx] == val):
                    if type(child_tree) is not dict:
                        class_label = child_tree
                    else:
                        class_label = predict_example(x, child_tree)

                    return class_label

        y_pred = sum([predict_example(x, h_l[1]) * h_l[0] for h_l in h_ens])
        y_pred /= sum([h_l[0] for h_l in h_ens])
        if y_pred > 0.5:
            y_pred = 1
        else:
            y_pred = 0
        return y_pred
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
    except Exception as e:
        print("An exception occurred:", e)


def confusion_matrix(y, y_pred, fig):
    confusion_matrix = np.zeros((2, 2))
    rows = ["Actual Positive", "Actual Negative"]
    cols = ("Classifier Positive", "Classifier Negative")
    for i in range(len(y_pred)):
        if y_pred[i] is not None and y[i] is not None:
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
             rowLabels=rows, colLabels=cols, loc="upper center")
    ax.axis("off")


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


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    print(decision_tree)
    y_pred = [predict_example_3(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print(f'Test error  = {(tst_err * 100):.2f}%.')

    """
    PART - 1 
    Construct four models for each combination of maximum depth d = 3,5 and bag size (k = 10,20). 
    Report the confusion matrix for these four settings
    """
    plt_count = 1
    for depth in [3, 5]:
        for bag_size in [10, 20]:
            h_ens_bagg = bagging(
                Xtrn, ytrn, max_depth=depth, num_trees=bag_size)
            y_pred_ens_bagg = [predict_example_ens(
                x, h_ens_bagg) for x in Xtst]
            tst_err_ens_bagg = compute_error(ytst, y_pred_ens_bagg)
            print(f'Test Error for Bagging = {tst_err_ens_bagg * 100:.2f} %, for depth = {depth} and stump = {bag_size}.')
            fig = plt.figure(plt_count)
            confusion_matrix(ytst, y_pred_ens_bagg, fig)
            fig.suptitle("Bagging: Max Depth " + str(depth) +
                         " and Bag Size " + str(bag_size))
            plt_count += 1

    """
    PART - 2
    Construct four models for each combination of maximum depth d = 1,2 and bag size (k = 20,40). 
    Report the confusion matrix for these four settings.
    """
    for depth in [1, 2]:
        for stump in [20, 40]:
            h_ens_boost = boosting(
                Xtrn, ytrn, max_depth=depth, num_stumps=stump)
            y_pred_ens_boost = [predict_example_ens(
                x, h_ens_boost) for x in Xtst]
            tst_err_ens_boost = compute_error(ytst, y_pred_ens_boost)
            print(f'Test Error for Boosting = {tst_err_ens_boost * 100:.2f} %, for depth = {depth} and stump = {stump}.')
            fig = plt.figure(plt_count)
            confusion_matrix(ytst, y_pred_ens_boost, fig)
            fig.suptitle("Boosting: Max Depth " + str(depth) +
                         " and Number of Stumps " + str(stump))
            plt_count += 1

    """
    PART - 3
    Use scikit-learn’s bagging and AdaBoost learners and repeat the experiments as described in parts (a) and (b) above. 
    Report the confusion matrices for these sets of settings. 
    What can you say about the quality of your implementation’s performance versus scikit’s performance?
    """
    # part - a Sklearn Bagging
    for depth in [3, 5]:
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        for bag_size in [10, 20]:
            sk_bagg = BaggingClassifier(estimator=dtree, n_estimators=bag_size)
            sk_bagg.fit(Xtrn, ytrn)
            y_pred_sk_bagg = sk_bagg.predict(Xtst)
            tst_err_sk_bagg = compute_error(ytst, y_pred_sk_bagg)
            print(f'Test Error for Sklearn Bagging = {tst_err_sk_bagg * 100 :.2f} %, for depth = {depth} and stump = {bag_size}.')
            fig = plt.figure(plt_count)
            confusion_matrix(ytst, y_pred_sk_bagg, fig)
            fig.suptitle("Sklearn Bagging: Max Depth " + str(depth) + " and Bag Size " + str(bag_size))
            plt_count += 1

    # part -b Sklearn Boosting
    for depth in [1, 2]:
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        for stump in [20, 40]:
            sk_boost = AdaBoostClassifier(estimator=dtree, n_estimators=stump)
            sk_boost.fit(Xtrn, ytrn)
            y_pred_sk_boost = sk_boost.predict(Xtst)
            tst_err_sk_boost = compute_error(ytst, y_pred_sk_boost)
            print(f'Test Error for Sklearn Boosting  = {tst_err_sk_boost * 100:.2f} %, for depth = {depth} and stump = {stump}.')
            fig = plt.figure(plt_count)
            confusion_matrix(ytst, y_pred_sk_boost, fig)
            fig.suptitle("Sklearn Boosting: Max Depth " + str(depth) + " and Number of Stumps " + str(stump))
            plt_count += 1

    # plt.show()