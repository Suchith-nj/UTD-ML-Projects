import numpy as np
# The "@" symbol in Python is used for matrix multiplication (also called the dot product). In this code, the "@" symbol is used to multiply the input "x" and the model weights "self.weights". This is equivalent to using the numpy function "np.dot(x, self.weights)".
"""
"""
class LogisticRegression:
    def __init__(self, w=None):
        self.w = w

    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def compute_loss(self, X, y):
        y_pred = self.sigmoid(X @ self.w)
        loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()
        return loss

    def gradient_ascent(self, X, y, lr):
        y_pred = self.sigmoid(X @ self.w)
        gradient = X.T @ (y_pred - y) / len(y)
        self.w -= lr * gradient

    def fit(self, X, y, lr=0.1, iters=100, recompute=True):
        if recompute or self.w is None:
            self.w = np.zeros(X.shape[1])

        for i in range(iters):
            self.gradient_ascent(X, y, lr)

    def predict_example(self, x):
        return int(self.sigmoid(x @ self.w) >= 0.5)

    def predict(self, X):
        return np.array([self.predict_example(x) for x in X])


def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


if __name__ == '__main__':
    train_file = './data/monks-3.train'
    test_file = './data/monks-3.test'

    train_data = np.loadtxt(train_file, delimiter=' ')
    test_data = np.loadtxt(test_file, delimiter=' ')

    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    for iter in [10, 100, 1000, 10000]:
        for a in [0.01, 0.1, 0.33]:
            model = LogisticRegression()
            model.fit(X_train, y_train, lr=a, iters=iter)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_error = compute_error(y_train, y_train_pred)
            test_error = compute_error(y_test, y_test_pred)

            print(f'Number of Iterations: {iter}, Learning Rate: {a}, Train Error: {train_error}, Test Error: {test_error}')
