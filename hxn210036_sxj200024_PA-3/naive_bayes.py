# naive_bayes.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 3 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the
# Logistic Regression algorithm. Insert your code into the various functions
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results.
#
# 4. Make sure to save your model in a pickle file after you fit your Naive
# Bayes algorithm.
#

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from collections import defaultdict
from heapq import nlargest
import matplotlib.pyplot as plt
import pprint
import math
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')


class Simple_NB():
    """
    A class for fitting the classical Multinomial Naive Bayes model that is especially suitable
    for text classifcation problems. Calculates the priors of the classes and the likelihood of each word
    occurring given a class of documents.
    """

    def __init__(self):
        # Instance variables for the class.
        self.priors = defaultdict(dict)
        self.likelihood = defaultdict(dict)
        self.columns = None

    def partition(self, x):
        """
        Partition the column vector into subsets indexed by its unique values (v1, ... vk)

        Returns a dictionary of the form
        { v1: indices of y == v1,
        v2: indices of y == v2,
        ...
        vk: indices of y == vk }, where [v1, ... vk] are all the unique values in the vector z.
        """
        try:
            unique_values = np.unique(x)
            dict_uni_values = {}
            for i in unique_values:
                dict_uni_values[i] = []
            for i in unique_values:
                dict_uni_values.update({i: np.where(x == i)[0]})
            return dict_uni_values
        except Exception as e: 
            print(e)

    def fit(self, X, y, column_names, alpha=1):
        """
        Compute the priors P(y=k), and the class-conditional probability (likelihood param) of each feature 
        given the class=k. P(y=k) is the the prior probability of each class k. It can be calculated by computing 
        the percentage of samples belonging to each class. P(x_i|y=k) is the number of counts feature x_i occured 
        divided by the total frequency of terms in class y=k.

        The parameters after performing smooothing will be represented as follows.
            P(x_i|y=k) = (count(x_i|y=k) + alpha)/(count(x|y=k) + |V|*alpha) 
            where |V| is the vocabulary of text classification problem or
            the size of the feature set

        :param x: features
        :param y: labels
        :param alpha: smoothing parameter

        Compute the two class instance variable 
        :param self.priors: = Dictionary | self.priors[label]
        :param self.likelihood: = Dictionary | self.likelihood[label][feature]

        """
        try:
            # INSERT CODE HERE
            y_priors = self.partition(y)
            y_labels = y_priors.keys()

            # Calculate prior probabilities for each class
            total_samples = len(y)
            for label in y_labels:
                samples_with_label = (y == label).sum()
                self.priors[label] = samples_with_label / total_samples

            # Initialize likelihood dictionary with 0 counts
            # for each feature in each label
            self.columns = column_names
            for label in y_labels:
                for column in column_names:
                    self.likelihood[label][column] = 0

            for i, row in enumerate(X):
                label = y[i]
                for j, count in enumerate(row):
                    column = column_names[j]
                    self.likelihood[label][column] += count

            # Perform smoothing to prevent zero probabilities
            vocab_size = len(column_names)
            for label in y_labels:
                total_count = sum(self.likelihood[label].values())
                for column in column_names:
                    count = self.likelihood[label][column]
                    self.likelihood[label][column] = (
                        count + alpha) / (total_count + alpha * vocab_size)

            # Add an extra key in your likelihood dictionary for unseen data
            # This will be used when testing sample texts that contain words not present in feature set
            for label in y_labels:
                self.likelihood[label]["_unseen_"] = alpha / \
                    (total_count + alpha * vocab_size)
                # print("priors", self.priors)
                # print("likelihood", self.likelihood)
        except Exception as e: 
            print(e)

    def predict_example(self, x, sample_text=False, return_likelihood=False):
        """
        Predicts the classification label for a single example x by computing the posterior probability
        for each class value, P(y=k|x) = P(x_i|y=k)*P(y=k).
        The predicted class will be the argmax of P(y=k|x) over all the different k's, 
        i.e. the class that gives the highest posterior probability
        NOTE: Converting the probabilities into log-space would help with any underflow errors.

        :param x: example to predict
        :return: predicted label for x
        :return: posterior log_likelihood of all classes if return_likelihood=True
        """
        try:
            loglikelihood = defaultdict(dict)
            for label in self.priors.keys():
                likelihoods = self.likelihood[label]
                logLHDict = {}
                for k in likelihoods.keys():
                    logLHDict[k] = np.log(likelihoods[k])
                loglikelihood[label] = logLHDict

            if return_likelihood:
                return loglikelihood

            if sample_text:
                frequencies = {}
                for word in x:
                    if word in frequencies:
                        frequencies[word] += 1
                    else:
                        frequencies[word] = 1
                word_freq = frequencies

                argm = np.zeros(2)
                for label in self.priors.keys():
                    likelihoods = self.likelihood[label]
                    prob = np.log(self.priors[label])
                    for word in word_freq.keys():
                        if word in self.columns:
                            llh_i = loglikelihood[label][word] * \
                                word_freq[word]
                        else:
                            llh_i = loglikelihood[label]["_unseen_"] * \
                                word_freq[word]
                        prob = prob + llh_i
                    argm[label] = prob
                return np.argmax(argm)
            else:
                zero_array = np.zeros(2)
                log_priors = {label: np.log(
                    self.priors[label]) for label in self.priors}
                log_unseen = {label: np.log(
                    self.likelihood[label]["_unseen_"]) for label in self.priors}
                for label in self.priors.keys():
                    likelihoods = self.likelihood[label]
                    prob = log_priors[label]
                    non_zero_arr = np.nonzero(x)[0]
                    for i in non_zero_arr:
                        k = self.columns[i]
                        if k in likelihoods:
                            loglikelihood_arr = loglikelihood[label][k] * x[i]
                        else:
                            loglikelihood_arr = log_unseen[label] * x[i]
                        prob += loglikelihood_arr
                    zero_array[label] = prob
                return np.argmax(zero_array)
        except Exception as e: 
            print(e)

    def getTopThreeWords(self):
        try:
            print(
                "The Top three words that have the highest class-conditional likelihoods for ")
            three_words_notspam = nlargest(
                3, self.likelihood[0], key=self.likelihood[0].get)
            print("'NotSpam'")
            for not_spam in three_words_notspam:
                print(not_spam, " is : ", self.likelihood[0].get(not_spam))

            three_words_spam = nlargest(
                3, self.likelihood[1], key=self.likelihood[1].get)
            print("'Spam'")
            for spam_val in three_words_spam:
                print(spam_val, " is : ", self.likelihood[1].get(spam_val))
        except Exception as e: 
            print(e)


def compute_accuracy(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    :param y_true: true label
    :param y_pred: predicted label
    :return: accuracy = 1/n * sum(y_true==y_pred)
    """
    try:
        n = len(y_true)
        accuracy = sum(y_true == y_pred) / n
        return accuracy
    except:
        raise Exception('Function not yet implemented!')


def compute_precision(y_true, y_pred):
    """
    Computes the precision for the given set of predictions.
    Precision gives the proportion of positive predictions that are actually correct. 
    :param y_true: true label
    :param y_pred: predicted label
    :return: precision
    """
    try:
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        precision = true_positives / (true_positives + false_positives)
        return precision
    except Exception as e: 
        print(e)


def compute_recall(y_true, y_pred):
    """
    Computes the recall for the given set of predictions.
    Recall measures the proportion of actual positives that were predicted correctly.
    :param y_true: true label
    :param y_pred: predicted label
    :return: recall

    """
    try:
        true_positives = sum((y_true == 1) & (y_pred == 1))
        false_negatives = sum((y_true == 1) & (y_pred == 0))
        recall = true_positives / (true_positives + false_negatives)
        return recall
    except Exception as e: 
        print(e)


def compute_f1(y_true, y_pred):
    """
    Computes the f1 score for a given set of predictions.
    F1 score is defined as the harmonic mean of precision and recall.
    :param y_true: true label
    :param y_pred: predicted label
    :return: f1 = 2 * (P*R)/(P+R)
    """
    try:
        precision = compute_precision(y_true, y_pred)
        recall = compute_recall(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    except Exception as e:
        print(e)


if __name__ == '__main__':

    df_train = pd.read_csv("data/train_email.csv")
    df_train.drop(df_train.columns[0], inplace=True, axis=1)

    df_test = pd.read_csv("data/test_email.csv")
    df_test.drop(df_test.columns[0], inplace=True, axis=1)

    X_columns = df_train.columns
    print(len(X_columns))
    print(df_train.shape)

    Xtrn = np.array(df_train.iloc[:, :-1])
    ytrn = np.array(df_train.iloc[:, -1])

    Xtst = np.array(df_test.iloc[:, :-1])
    ytst = np.array(df_test.iloc[:, -1])
    results = {}  # To Store All the Results

    # # PART A
    NB = Simple_NB()
    NB.fit(Xtrn, ytrn, column_names=X_columns, alpha=1)

    # Prediction on Test Set
    y_pred = []
    for x in Xtst:
        y_pred.append(NB.predict_example(x))
    y_pred = np.asarray(y_pred)

    accuracy = compute_accuracy(ytst, y_pred)
    precision = compute_precision(ytst, y_pred)
    recall = compute_recall(ytst, y_pred)
    f1_scr = compute_f1(ytst, y_pred)

    results["Simple Naive Bayes"] = {"accuracy": accuracy,
                                         "precision": precision,
                                         "recall": recall,
                                         "f1_score": f1_scr,
                                         }
    print(results)
    top_three_words = NB.getTopThreeWords()
    # print(top_three_words)

    # PART B - Testing on Sample Text

    sample_email = ["Congratulations! Your raffle ticket has won yourself a house. Click on the link to avail prize",
                    "Hello. This email is to remind you that your project needs to be submitted this week"]

    for sample in sample_email:
        words = nltk.word_tokenize(sample)
        tokens = [word.lower() for word in words if word.isalpha()]
        y_sent_pred = NB.predict_example(tokens, sample_text=True)
        print("Sample Email: {} \nIs Spam".format(sample)) if y_sent_pred else print(
            "Sample Text: {} \nIs Not Spam".format(sample))

    # PART C - Compare with Sklearn's NB Models
    # Replace Nones with the respective Sklearn library.
    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Multimodel Naive Bayes": MultinomialNB(),
        "Bernoulli Naive Bayes": BernoulliNB()
    }
    acc_scores, prec_scores, recall_scores, f1_scores = {}, {}, {}, {}

    for model_name, sk_lib in models.items():

        model = sk_lib
        model.fit(Xtrn, ytrn)

        # Predict the target values for test set
        y_pred = model.predict(Xtst)
        accuracy = accuracy_score(ytst, y_pred)
        precision = precision_score(ytst, y_pred, average='weighted')
        recall = recall_score(ytst, y_pred, average='weighted')
        f1 = f1_score(ytst, y_pred, average='weighted')

        results[model_name] = {"accuracy": accuracy,
                               "precision": precision,
                               "recall": recall,
                               "f1_score": f1
                               }
        acc_scores[model_name] = accuracy
        prec_scores[model_name] = precision
        recall_scores[model_name] = recall
        f1_scores[model_name] = f1
    # results = results.update(results_SNB)

    pprint.pprint(results)
    # Based on these results, the Gaussian Naive Bayes model has the highest accuracy, precision score, indicating that
    # it is the most accurate in correctly identifying legitimate emails as not spam. However, it is important
    # to consider other metrics as well, such as recall and F1 score, to make a more informed decision on which model to use.

    # PART D - Visualize the model using bar charts
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['blue', 'orange', 'green', 'red']
    fig, axs = plt.subplots(1, len(metrics), figsize=(16, 6))

    width = 0.2  # width of the bars

    for i, metric in enumerate(metrics):
        x = np.arange(len(results))

        bars = axs[i].bar(x - width, [results[model][metric] for model in results], width=width, label=metric.capitalize())

        axs[i].set_xlabel('Naive Bayes Models')
        axs[i].set_ylabel(metric.capitalize())

        for j, bar in enumerate(bars):
            bar.set_color(colors[j])
            axs[i].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, format(bar.get_height(), '.2f'), ha='center', va='bottom')

    fig.legend(bars, [model for model in results], loc='lower center', ncol=4)

    
    for j, color in enumerate(colors):
        plt.text(j * 0.25, -0.1, f"{metrics[j].capitalize()}: {format([results[model][metrics[j]] for model in results][j], '.2f')}", color=color)

    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.show()


    # PART E - Save the Model
    netid = 'hxn210036_sxj200024'
    with open(f'{netid}_model_1.obj', 'wb') as file_pi:
        pickle.dump(NB, file_pi)
