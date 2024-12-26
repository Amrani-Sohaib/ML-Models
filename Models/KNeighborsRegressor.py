import numpy as np

class KNearestNeighborsRegressor:
    def __init__(self, k=3):
        """
        Initialize the k-NN regressor.

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
        Predict the target values for the given data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
        np.ndarray: Predicted target values of shape (n_samples,)
        """
        predictions = [self._predict_single_point(x) for x in X]
        return np.array(predictions)

    def _predict_single_point(self, x):
        """
        Predict the target value for a single data point.

        Parameters:
        x (np.ndarray): Data point to predict.

        Returns:
        float: Predicted target value (average of k nearest neighbors).
        """
        # Compute distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the target values of the k nearest neighbors
        k_neighbor_values = [self.y_train[i] for i in k_indices]

        # Return the average of the k nearest neighbors' target values
        return np.mean(k_neighbor_values)

# Example usage
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    y_train = np.array([1.5, 2.0, 3.5, 5.5])

    X_test = np.array([[2, 3], [4, 5]])

    # Initialize k-NN regressor with k=3
    knn_regressor = KNearestNeighborsRegressor(k=3)

    # Fit the model
    knn_regressor.fit(X_train, y_train)

    # Predict on test data
    predictions = knn_regressor.predict(X_test)
    print("Predictions:", predictions)
