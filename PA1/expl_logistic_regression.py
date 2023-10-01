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
        #DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w

    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        # INSERT YOUR CODE HERE
        """
        def compute_loss(self, X, y):
            w = self.weights
            m = X.shape[0]
            z = np.dot(X, w)
            h = 1 / (1 + np.exp(-z))
            loss = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
            return loss

        """
        """
        In this implementation,
         the dot product of the feature matrix X and the model weights w is calculated
        to get the logits z. Then the logistic function (also known as the sigmoid function)
         is applied to the logits to get the predicted probabilities h. 
         Finally, the binary cross-entropy loss is calculated as the average of the 
         loss for each sample in the batch, which is the sum of the negative 
         log-likelihood of the true label.

        """

        raise Exception('Function not yet implemented!')

    
    def sigmoid(self, val):

        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        # INSERT YOUR CODE HERE
        """
        def sigmoid(self, val):
            return 1.0 / (1.0 + np.exp(-val))
        """    
        """
        This function takes a float or a numpy array as input, and returns the sigmoid of the input value. The sigmoid function maps any real-valued number to the range of [0, 1]. In logistic regression, it is used as the activation function to transform the dot product of the weights and features into a probability of the positive class.

        """
        raise Exception('Function not yet implemented!')


    def gradient_ascent(self, w, X, y, lr):

        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        # INSERT YOUR CODE HERE
        """
        def gradient_ascent(self, w, X, y, lr):
            m, n = X.shape
            y_pred = self.sigmoid(np.dot(X, w ))
            error = y_pred - y
            gradient = np.dot(X.T, error) / m
            w = w + lr * gradient
            return w
        """

        """
        In this function,
         we first calculate the predicted output y_pred using
          the sigmoid function and the current weights w.
           We then compute the error between the predicted output
            and actual output y. We then calculate the gradient of 
            the loss with respect to the weights w. 
            Finally, we update the weights w using the gradient 
            ascent algorithm by adding the product of learning rate lr 
            and the gradient to the current weights.

        """
        raise Exception('Function not yet implemented!')



    def fit(self,X, y, lr=0.1, iters=100, recompute=True):
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
        
        if(recompute):
            #Reinitialize the model weights
            # self.weights = np.zeros(X.shape[1])
            pass

        for _ in range(iters):
            # INSERT YOUR CODE HERE
            """
            for i in range(iters):
                z = np.dot(X, self.weights)
                y_hat = self.sigmoid(z)
                gradient = np.dot(X.T, y_hat - y) / y.size
                self.weights -= lr * gradient
                """
            pass

            """
            In this implementation, the fit function takes as input the features X, the label y, the learning rate lr, the number of iterations iters, and the recompute flag. If recompute is set to True, the function reinitializes the weights to zeros. Then, in a loop, the function computes the predicted label y_hat using the sigmoid function, calculates the gradient of the binary cross-entropy loss using y_hat and y, and updates the weights using gradient ascent.

            """

    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
         # INSERT YOUR CODE HERE

        """
        z = np.dot(w, x)
        prediction = self.sigmoid(z)
        
        if prediction >= 0.5:
            return 1
        else:
            return 0
        """ 
        raise Exception('Function not yet implemented!')



    def compute_error(y_true, y_pred):
        """
        Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
        :param y_true: true label
        :param y_pred: predicted label
        :return: error rate = (1/n) * sum(y_true!=y_pred)
        """
        # INSERT YOUR CODE HERE
        """
        n = len(y_true)
        error = sum(y_true != y_pred) / n
        return error
        """
        raise Exception('Function not yet implemented!')




if __name__ == '__main__':

    # Load the training data
    M = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    lr =  SimpleLogisiticRegression()
    
    #Part 1) Compute Train and Test Errors for different number of iterations and learning rates
    for iter in [10, 100,1000,10000]:
        for a in [0.01,0.1, 0.33]:
            #INSERT CODE HERE
            pass

    #Part 2) Retrain Logistic Regression on the best parameters and store the model as a pickle file
    #INSERT CODE HERE

    """
    import pickle

    best_iter = 10000
    best_lr = 0.33

    # retrain the model with best parameters
    lr = LogisticRegression(best_lr)
    lr.fit(X_train, y_train, lr=best_lr, iters=best_iter)

    # store the retrained model as a pickle file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(lr, f)

    """
    # Code to store as pickle file
    netid = ''
    file_pi = open('{}_model_1.obj',format(netid), 'wb')  #Use your NETID
    pickle.dump(lr, file_pi)


    #Part 3) Compare your model's performance to scikit-learn's LR model's default parameters 
    #INSERT CODE HERE

    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Load the train data into a numpy array
    train_data = np.loadtxt('./data/monks-3.train')
    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]

    # Load the test data into a numpy array
    test_data = np.loadtxt('./data/monks-3.test')
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]

    # Fit the scikit-learn logistic regression model
    scikit_lr = LogisticRegression()
    scikit_lr.fit(train_features, train_labels)

    # Predict the labels for the test set
    scikit_preds = scikit_lr.predict(test_features)

    # Compute the accuracy of the scikit-learn model
    scikit_acc = accuracy_score(test_labels, scikit_preds)

    # Load your custom logistic regression model from the pickle file
    # Replace "my_lr_model.pickle" with the name of your pickle file
    import pickle
    with open('my_lr_model.pickle', 'rb') as f:
        my_lr = pickle.load(f)

    # Predict the labels for the test set using your custom model
    my_preds = my_lr.predict(test_features)

    # Compute the accuracy of your custom model
    my_acc = accuracy_score(test_labels, my_preds)

    # Compare the accuracy of the two models
    print("Scikit-learn LR Model Accuracy:", scikit_acc)
    print("Custom LR Model Accuracy:", my_acc)

    """

    """
    In this code, we first load the train and test data into numpy arrays using numpy.loadtxt(). We then fit a logistic regression model from scikit-learn and use it to predict the labels for the test set. The accuracy of the model is computed using sklearn.metrics.accuracy_score().

    Next, we load the custom logistic regression model that we trained earlier from a pickle file and use it to make predictions on the test set. Finally, we compare the accuracy of the two models.

    """

    #Part 4) Plot curves on train and test loss for different learning rates. Using recompute=False might help
    for a in [0.01,0.1, 0.33]:
        lr.fit(Xtrn, ytrn, lr=a, iters=1)
        #INSERT CODE HERE
        for i in range(10):
            lr.fit(Xtrn, ytrn, lr=a, iters=100,recompute=False)
            #INSERT CODE HERE


    """
    import matplotlib.pyplot as plt

    for a in [0.01, 0.1, 0.33]:
        lr.fit(Xtrn, ytrn, lr=a, iters=1)
        train_loss = []
        test_loss = []
        for i in range(10):
            lr.fit(Xtrn, ytrn, lr=a, iters=100, recompute=False)
            train_pred = lr.predict(Xtrn)
            test_pred = lr.predict(Xtst)
            train_loss.append(compute_error(ytrn, train_pred))
            test_loss.append(compute_error(ytst, test_pred))
        plt.plot(range(10), train_loss, label="train, lr={}".format(a))
        plt.plot(range(10), test_loss, label="test, lr={}".format(a))

    plt.xlabel("Iteration")
    plt.ylabel("Error rate")
    plt.legend()
    plt.show()

    """


