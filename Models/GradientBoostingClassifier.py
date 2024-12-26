import numpy as np

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        """
        Initialize the Gradient Boosting Classifier.

        Parameters:
        n_estimators (int): Number of boosting rounds (weak learners).
        learning_rate (float): Step size for weight updates.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.initial_prediction = None

    def _compute_residuals(self, y, y_pred):
        """
        Compute the residuals for classification as the negative gradient.

        Parameters:
        y (np.ndarray): True class labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities.

        Returns:
        np.ndarray: Residuals.
        """
        return y - y_pred

    def _sigmoid(self, z):
        """
        Apply the sigmoid function.

        Parameters:
        z (np.ndarray): Input array.

        Returns:
        np.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the Gradient Boosting Classifier to the training data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary target vector of shape (n_samples,), where labels are 0 or 1.
        """
        # Initialize predictions to the log-odds of the positive class
        y_mean = np.mean(y)
        self.initial_prediction = np.log(y_mean / (1 - y_mean))
        y_pred = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            # Compute probabilities and residuals
            probabilities = self._sigmoid(y_pred)
            residuals = self._compute_residuals(y, probabilities)

            # Train a simple regression tree (stump) on the residuals
            tree = DecisionStumpRegressor()
            tree.fit(X, residuals)
            residual_pred = tree.predict(X)

            # Update predictions
            y_pred += self.learning_rate * residual_pred

            # Store the weak learner
            self.models.append(tree)

    def predict(self, X):
        """
        Predict class labels for the given data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,).
        """
        # Start with the initial prediction
        y_pred = np.full(X.shape[0], self.initial_prediction)

        # Add contributions from all trees
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)

        # Convert to class labels
        probabilities = self._sigmoid(y_pred)
        return (probabilities >= 0.5).astype(int)

class DecisionStumpRegressor:
    def __init__(self):
        """
        Initialize a simple decision stump for regression.
        """
        self.feature_index = None
        self.threshold = None
        self.output_left = None
        self.output_right = None

    def fit(self, X, y):
        """
        Fit the decision stump to the data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vector of shape (n_samples,).
        """
        best_loss = float('inf')

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                output_left = np.mean(y[left_mask]) if np.any(left_mask) else 0
                output_right = np.mean(y[right_mask]) if np.any(right_mask) else 0

                predictions = np.zeros(y.shape)
                predictions[left_mask] = output_left
                predictions[right_mask] = output_right

                loss = np.mean((y - predictions) ** 2)

                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.output_left = output_left
                    self.output_right = output_right

    def predict(self, X):
        """
        Predict target values using the decision stump.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted target values of shape (n_samples,).
        """
        predictions = np.zeros(X.shape[0])
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = X[:, self.feature_index] > self.threshold

        predictions[left_mask] = self.output_left
        predictions[right_mask] = self.output_right

        return predictions

# Example usage
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])

    X_test = np.array([[1.5, 2.5], [3.5, 4.5], [4.5, 5.5]])

    # Initialize the Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)

    # Fit the model
    gbc.fit(X_train, y_train)

    # Predict on test data
    predictions = gbc.predict(X_test)
    print("Predictions:", predictions)
