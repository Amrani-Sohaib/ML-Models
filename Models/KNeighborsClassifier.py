import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Initialize the k-NN classifier.

        Parameters:
        k (int): Number of neighbors to consider.
        """
        self.k = k

    def fit(self, X, y):
        """
        Store the training data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two points.

        Parameters:
        x1 (np.ndarray): First point.
        x2 (np.ndarray): Second point.

        Returns:
        float: Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

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
        Predict the label for a single data point.

        Parameters:
        x (np.ndarray): Data point to predict.

        Returns:
        int or str: Predicted class label.
        """
        # Compute distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbors
        k_neighbor_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

# Example usage
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    y_train = np.array([0, 0, 1, 1])

    X_test = np.array([[2, 3], [4, 5]])

    # Initialize k-NN with k=3
    knn = KNearestNeighbors(k=3)

    # Fit the model
    knn.fit(X_train, y_train)

    # Predict on test data
    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
