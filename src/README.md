## Source Code
This directory contains the source code for the `NewTrainer` and `LinearRegression` classes.

### `new_trainer.py`
The `NewTrainer` class provides an abstract interface for training and predicting on a machine learning problem. Any new machine learning algorithm can extend the `NewTrainer` class and implement its `train` and `predict` methods.

### `linear_regression.py`
The `LinearRegression` class is an implementation of linear regression using the `NewTrainer` abstract class as a reference implementation. It provides an implementation for training and predicting on a linear regression problem as illustrated in Linear RegressionExample problem in next section.

## Linear Regression
This module contains an implementation of linear regression, a popular machine learning algorithm for predicting a numerical output based on one or more input variables.


### Example Problem
As an example, suppose we want to predict the price of a house based on its size (in square feet) and the number of bedrooms it has. We have a dataset of houses with known sizes and numbers of bedrooms, along with their corresponding prices. Our goal is to train a linear regression model on this dataset so that we can predict the price of a new house given its `size` and `number of bedrooms`.

To solve this problem, we can use the LinearRegression class provided in `linear_regression.py`. We will pass in a list of houses, where each house is represented as a list of its size and number of bedrooms. We will also pass in a list of prices corresponding to each house. Here is an example of how we might use this class:
```python
from linear_regression import LinearRegression

# Create a new linear regression model
model = LinearRegression()

# Define our training data
houses = [[1400, 3], [1600, 3], [1700, 2], [1875, 3], [1100, 2], [1550, 4], [2350, 4], [2450, 4], [1425, 3], [1700, 3]]
prices = [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]

# Train the model on our data
model.train(houses, prices)

# Use the model to predict the price of a new house
new_house = [2000, 3]
predicted_price = model.predict(new_house)
print(f"Predicted price for new house: ${predicted_price:.2f}")
# Predicted price for new house: $320597.12
```

In this example, we first create a new instance of the `LinearRegression` class. We then define our training data as two lists: `houses` and `prices`. We call the train method on our model instance, passing in these two lists as arguments. This trains the model on our data.

We can then use the `predict` method on our trained model to make predictions for new houses. In this case, we create a new house with 2000 square feet and 3 bedrooms, and pass this as an argument to the `predict` method. The model returns a predicted price of `$320597.12` for this new house.
