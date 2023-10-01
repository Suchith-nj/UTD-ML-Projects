# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
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
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results.


import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle

class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the model weights
    """

    def __init__(self):
        self.w = []
        pass

    def initialize_weights(self, num_features):
        # DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w

    def sigmoid(self, val):
        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        # INSERT YOUR CODE HERE
        try:
            result = 1.0 / (1.0 + np.exp(-val))
            return result
        except Exception as e:
            print(e)

    def gradient_ascent(self, w, X, y, lr):
        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        try:
            z = np.dot(X, w)
            # Apply the sigmoid function to get the predicted probabilities
            y_pred = self.sigmoid(z)
            # Compute the gradient of the loss function with respect to the weights
            gradient = np.dot(X.T, (y - y_pred))
            # Update the model weights
            w += lr * gradient
            return w
        except Exception as e:
            print(e)

    def fit(self, X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """
        # INSERT YOUR CODE HERE
        no_of_features = X.shape[1]

        if (recompute):
            # Reinitialize the model weights
            self.w = np.zeros(no_of_features)

        for _ in range(iters):
            # INSERT YOUR CODE HERE
            z = np.dot(X, self.w)
            y_pred = self.sigmoid(z)
            dw = np.dot(X.T, (y_pred - y)) / X.shape[0]
            self.w -= lr * dw


    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
        # INSERT YOUR CODE HERE
        try:
            z = np.dot(x, w)
            y_pred = self.sigmoid(z)
            return (y_pred >= 0.5).astype(int)
        except Exception as e:
            print(e)

    def compute_error(self, y_true, y_pred):
        """
        Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
        :param y_true: true label
        :param y_pred: predicted label
        :return: error rate = (1/n) * sum(y_true!=y_pred)
        """
        # INSERT YOUR CODE HERE
        try:
            error_rate = np.mean(y_true != y_pred)
            return error_rate
        except Exception as e:
            print(e)
    
    def compute_loss(self, X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        # INSERT YOUR CODE HERE
        try:
            w = self.w
            m = X.shape[0]
            z = np.dot(X, w)
            h = self.sigmoid(z)
            loss = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
            return loss
        except Exception as e:
            print(e)


if __name__ == '__main__':

    # Load the training data 
    MTrain = np.genfromtxt('./data/monks-3.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = MTrain[:, 0]
    Xtrn = MTrain[:, 1:]

    # Load the test data
    MTest = np.genfromtxt('./data/monks-3.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = MTest[:, 0]
    Xtst = MTest[:, 1:]

    lr = SimpleLogisiticRegression()

    results = []
    # Part 1) Compute Train and Test Errors for different number of iterations and learning rates
    for iter in [10, 100, 1000, 10000]:
        for a in [0.01, 0.1, 0.33]:
            # INSERT CODE HERE
            lr.fit(Xtrn, ytrn, lr=a, iters=iter)
            weights = lr.w
            train_pred = lr.predict_example(weights, Xtrn)
            test_pred = lr.predict_example(weights, Xtst)
            train_error = lr.compute_error(ytrn, train_pred)
            test_error = lr.compute_error(ytst, test_pred)
            results.append((iter, a, train_error, test_error))

    # Print the results
    # print("iterations, learning rate, train error, test error")
    # for res in results:
    #     print(f"{results}\n")
    # Reporting best Parameters found
    min_test_error = min(results, key=lambda x: x[3])[3]
    best_iter, best_lr = min(results, key=lambda x: x[3])[0], min(results, key=lambda x: x[3])[1]
    print("__________________________________________________________________")
    print(f"Minimum test error: {min_test_error}")
    print("__________________________________________________________________")
    print(f"Best Parametersfor our model: \n Best iterations:{best_iter} \n Best Learning Rate: {best_lr}" )
    print("__________________________________________________________________")

    # Part 2) Retrain Logistic Regression on the best parameters and store the model as a pickle file
    # INSERT CODE HERE

    # # retrain the model with best parameters
    lr_best_model = SimpleLogisiticRegression()
    lr_best_model.fit(Xtrn, ytrn, lr=best_lr, iters=best_iter)

    # Code to store as pickle file
    netid = 'sxj200024'
    with open(f'{netid}_model_1.obj', 'wb') as file_pi:
        pickle.dump(lr_best_model,file_pi)

    # Part 3) Compare your model's performance to scikit-learn's LR model's default parameters
    # INSERT CODE HERE

    X_train, y_train = MTrain[:, 1:], MTrain[:, 0]
    X_test, y_test = MTest[:, 1:], MTest[:, 0]

    scikit_results = []

    scikit_lr = LogisticRegression()
    scikit_lr.fit(X_train, y_train)

    y_train_pred = scikit_lr.predict(X_train)
    y_test_pred = scikit_lr.predict(X_test)

    scikit_trn_error = lr.compute_error(y_train, y_train_pred)
    scikit_tst_error = lr.compute_error(y_test, y_test_pred)
    
    # compare the scikit train error and test error with part-1 best parameters errors

    print("scikit error rates:")
    print(f' scikit train error: {scikit_trn_error} \n scikit test error:{scikit_tst_error}')

    y_best_train_pred = lr_best_model.predict_example(lr_best_model.w,X_train)
    y_best_test_pred = lr_best_model.predict_example(lr_best_model.w, X_test)

    best_trn_error = lr_best_model.compute_error(y_train, y_best_train_pred)
    best_tst_error = lr_best_model.compute_error(y_test, y_best_test_pred)

    print("__________________________________________________________________")
    print("Error rates of our Model with best parameter:")
    print(f' our train error: {best_trn_error} \n our test error:{best_tst_error}')


    # Part 4) Plot curves on train and test loss for different learning rates. Using recompute=False might help
    for x, a in enumerate([0.01, 0.1, 0.33]):
        lr.fit(Xtrn, ytrn, lr=a, iters=1)
        # INSERT CODE HERE
        train_loss = []
        test_loss = []
        for i in range(11):
            lr.fit(Xtrn, ytrn, lr=a, iters=100, recompute=False)
            weights = lr.w
            # INSERT CODE HERE
            train_pred = lr.predict_example(weights, Xtrn)
            test_pred = lr.predict_example(weights, Xtst)
            train_loss.append(lr.compute_loss(Xtrn, ytrn))
            test_loss.append(lr.compute_loss(Xtst, ytst))
        print("__________________________________________________________________")
        print(f"For learning rate {a}")
        print(f"Training Loss:{train_loss}")
        print(f"Test Loss:{test_loss}")

        epochs = np.arange(0, 1001, 100)
        plt.subplot(1, 3, x+1)
        plt.plot(range(0, 1100, 100), train_loss, 'r', label='Training Loss')
        plt.plot(range(0, 1100, 100), test_loss, 'b', label='Test Loss')
        plt.title('Learning Rate: {}'.format(a))
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.legend()

plt.show()