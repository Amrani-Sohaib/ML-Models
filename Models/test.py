from Models.knn_regressor import KNearestNeighborsRegressor
import numpy as np


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
