import numpy as np

class GaussianNaiveBayesRegressor:
    def __init__(self):
        """
        Initialize the Gaussian Naive Bayes regressor.
        """
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target vector of shape (n_samples,)
        """
        self.mean = X.mean(axis=0)
        self.var = X.var(axis=0)
        self.y_mean = y.mean()

    def _gaussian_pdf(self, x):
        """
        Compute the Gaussian probability density function.

        Parameters:
        x (np.ndarray): Feature vector.

        Returns:
        np.ndarray: Probability density for each feature.
        """
        numerator = np.exp(-((x - self.mean) ** 2) / (2 * self.var))
        denominator = np.sqrt(2 * np.pi * self.var)
        return numerator / denominator

    def predict(self, X):
        """
        Predict the target values for the given data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
        np.ndarray: Predicted target values of shape (n_samples,)
        """
        probabilities = self._gaussian_pdf(X)
        return probabilities * self.y_mean

# Example usage
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [6.0, 8.0], [7.0, 8.0], [8.0, 9.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[2.0, 2.5], [7.5, 8.5]])

    # Initialize the Naive Bayes regressor
    gnb_regressor = GaussianNaiveBayesRegressor()

    # Fit the model
    gnb_regressor.fit(X_train, y_train)

    # Predict on test data
    predictions = gnb_regressor.predict(X_test)
    print("Predictions:", predictions)
