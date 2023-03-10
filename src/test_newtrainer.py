import pytest
import numpy as np
from typing import List
from newtrainer import LinearRegression

@pytest.fixture
def linear_regression_model():
    model = LinearRegression()
    return model

def test_train(linear_regression_model):
    x = [[1], [2], [3], [4], [5]]
    y = [2, 4, 6, 8, 10]
    linear_regression_model.train(x, y)
    assert np.allclose(linear_regression_model.coef_, [0, 2])

def test_train_empty_input(linear_regression_model):
    x = []
    y = []
    with pytest.raises(ValueError):
        linear_regression_model.train(x, y)

def test_train_input_length_mismatch(linear_regression_model):
    x = [[1], [2], [3], [4], [5]]
    y = [2, 4, 6, 8]
    with pytest.raises(ValueError):
        linear_regression_model.train(x, y)

def test_predict(linear_regression_model):
    x = [6]
    y = [12]
    linear_regression_model.train([[1], [2], [3], [4], [5]], [2, 4, 6, 8, 10])
    assert np.isclose(linear_regression_model.predict(x), y)

def test_predict_empty_input(linear_regression_model):
    x = []
    with pytest.raises(ValueError):
        linear_regression_model.predict(x)

def test_predict_input_length_mismatch(linear_regression_model):
    x = [1, 6]
    with pytest.raises(ValueError):
        linear_regression_model.predict(x)

