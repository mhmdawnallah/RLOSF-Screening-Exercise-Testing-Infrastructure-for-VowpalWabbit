"""
This module contains the Linear Regression which is the implementation of the abstract class NewTrainer
"""

from typing import List
import numpy as np
from new_trainer import NewTrainer

class LinearRegression(NewTrainer):
    def __init__(self):
        """
        Initializes an instance of the LinearRegression class with coef_ set to None.
        """
        self.coef_ = None

    def train(self, x: List[List[float]], y: List[float]):
        """
        Fits the linear regression model using the input data and updates the coefficients.

        Args:
            x: A list of lists containing the training input data. Each sublist represents a single observation and should contain the input features.
            y: A list containing the target variable values.

        Raises:
            ValueError: If the length of x is not equal to the length of y or if either x or y is empty.
        """
        if len(x) != len(y):
            raise ValueError("Input arrays must have the same length")
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays must not be empty")

        x = np.insert(x, 0, 1, axis=1)
        x_transpose = x.T
        x_transpose_x = np.dot(x_transpose, x)
        x_transpose_x_inv = np.linalg.inv(x_transpose_x)
        x_transpose_y = np.dot(x_transpose, y)
        self.coef_ = np.dot(x_transpose_x_inv, x_transpose_y)

    def predict(self, x: List[float]) -> float:
        """
        Predicts the target variable value for a given input.

        Args:
            x: A list containing the input features for which to predict the target variable value.

        Returns:
            The predicted target variable value.

        Raises:
            ValueError: If the model has not been trained yet or if the length of x does not match the length of the coefficient vector.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been trained yet")
        if len(x) != len(self.coef_) - 1:
            raise ValueError("Input length does not match coefficient length")

        x_with_intercept = [1] + x
        return np.dot(self.coef_, x_with_intercept)
