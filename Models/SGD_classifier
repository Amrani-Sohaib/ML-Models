import numpy as np

class SGDClassifier:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iter):
            for i in range(X.shape[0]):
                linear_output = np.dot(X[i], self.w) + self.b
                y_pred = self._sigmoid(linear_output)
                error = y[i] - y_pred

                self.w += self.learning_rate * error * X[i]
                self.b += self.learning_rate * error

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(linear_output)
        return np.where(y_pred >= 0.5, 1, 0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    # Initialize and train the classifier
    clf = SGDClassifier(learning_rate=0.01, n_iter=1000)
    clf.fit(X, y)

    # Make predictions
    predictions = clf.predict(X)
    print(predictions)