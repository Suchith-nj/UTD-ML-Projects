import numpy as np

class LogisticRegression:

    def sigmoid(self, val):
        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        return 1/(1 + np.exp(-val))
    
    def compute_loss(self, w, X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        m = len(y)
        h = self.sigmoid(np.dot(X, w))
        loss = (-1/m)*(np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        return loss
    
    def gradient_ascent(self, w, X, y, lr):
        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        m = len(y)
        h = self.sigmoid(np.dot(X, w))
        gradient = (1/m)*np.dot(X.T, (h-y))
        w = w - lr*gradient
        return w
    
    def fit(self, X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True
        """
        if recompute:
            self.weights = np.zeros(X.shape[1])
        else:
            self.weights = self.weights
        
        for i in range(iters):
            self.weights = self.gradient_ascent(self.weights, X, y, lr)
            
    def predict_example(self, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
        z = np.dot(x, self.weights)
        prediction = self.sigmoid(z)
        return 1 if prediction >= 0.5 else 0
    
    def predict(self, X):
        """
        Predicts the classification label for multiple examples using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param X: features
        :return: predicted labels for X
        """
        # predictions = [self.predict_example(x) for x in X
        pass