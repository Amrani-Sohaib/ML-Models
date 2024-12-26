import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        """
        Initialize the Gaussian Naive Bayes classifier.
        """
        self.classes = None
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
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.mean = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.var = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)

        for idx, cls in enumerate(self.classes):
            X_c = X[y == cls]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def _gaussian_pdf(self, class_idx, x):
        """
        Compute the Gaussian probability density function for a given class.

        Parameters:
        class_idx (int): Index of the class.
        x (np.ndarray): Feature vector.

        Returns:
        np.ndarray: Probability density for each feature.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,)
        """
        predictions = [self._predict_single_point(x) for x in X]
        return np.array(predictions)

    def _predict_single_point(self, x):
        """
        Predict the class label for a single data point.

        Parameters:
        x (np.ndarray): Feature vector.

        Returns:
        int: Predicted class label.
        """
        posteriors = []

        for idx, cls in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            conditional = np.sum(np.log(self._gaussian_pdf(idx, x)))
            posterior = prior + conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

# Example usage
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [6.0, 8.0], [7.0, 8.0], [8.0, 9.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[2.0, 2.5], [7.5, 8.5]])

    # Initialize the Naive Bayes classifier
    gnb = GaussianNaiveBayes()

    # Fit the model
    gnb.fit(X_train, y_train)

    # Predict on test data
    predictions = gnb.predict(X_test)
    print("Predictions:", predictions)
