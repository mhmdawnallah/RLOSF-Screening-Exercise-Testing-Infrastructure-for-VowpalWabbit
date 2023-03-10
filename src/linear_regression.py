from typing import List
import numpy as np
from new_trainer import NewTrainer

class LinearRegression(NewTrainer):
    def __init__(self):
        self.coef_ = None

    def train(self, x: List[List[float]], y: List[float]):
        if len(x) != len(y):
            raise ValueError("Input arrays must have the same length")
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays must not be empty")

        # Add a column of ones to x for the intercept term
        x = np.insert(x, 0, 1, axis=1)

        # Compute the coefficients using the normal equation
        x_transpose = x.T
        x_transpose_x = np.dot(x_transpose, x)
        x_transpose_x_inv = np.linalg.inv(x_transpose_x)
        x_transpose_y = np.dot(x_transpose, y)
        self.coef_ = np.dot(x_transpose_x_inv, x_transpose_y)

    def predict(self, x: List[float]) -> float:
        if self.coef_ is None:
            raise ValueError("Model has not been trained yet")
        if len(x) != len(self.coef_) - 1:
            raise ValueError("Input length does not match coefficient length")

        # Add a 1 to the beginning of x for the intercept term
        x_with_intercept = [1] + x
        return np.dot(self.coef_, x_with_intercept)
