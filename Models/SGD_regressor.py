import numpy as np

class SGD_Regressor:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for idx in range(n_samples):
                linear_model = np.dot(X[idx], self.weights) + self.bias
                error = y[idx] - linear_model

                self.weights += self.learning_rate * error * X[idx]
                self.bias += self.learning_rate * error

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

