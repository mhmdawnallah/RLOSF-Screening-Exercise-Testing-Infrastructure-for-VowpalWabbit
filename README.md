# RL Open Source Fest Screening Exercise 2023

## General Description

This repository contains pytests for a new training algorithm for regression with the `NewTrainer` abstract interface and Its implementation `LinearRegression` class on completing the screening exercise of "Testing infrastructure for VowpalWabbit" RLOSF 2023 Project.

Submitted by: Mohamed Awnallah <mohamedmohey2352@gmail.com> <br>

# Description of the Screening exercise
Let’s say we have just implemented a new training algorithm for regression with the following interface:

```python
class NewTrainer:
    ...
    def train(self, x: List[List[float]], y: List[float]):
        ...

    def predict(self, x: List[float]) -> float:
        ...
        return 0
```
```
Design and write test suite for it in Python using unittest or pytest frameworks.
```
## Requirements
The project requires the following dependencies:
- numpy
To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Running Tests
To run the tests, use the following command from the root directory:
```bash
pytest -v
```

## CI/CD Workflow
Although CI/CD (Continous Integration/Continous Deployment) workflow is not required in the screening exercise in `Testing Infrastructure for VowalWabbit`, I'd like to take it a step further and integrate the testing of the linear regression machine learning with GitHub actions. Additionally, GitHub pages will be used to display [Allure](https://github.com/allure-framework/allure-python) testing report and README documentation.

The latest updated testing report could be found [here](https://mhmdawnallah.github.io/RLOSF-Screening-Exercise-Testing-Infrastructure-for-VowalWabbit/allure_testing_report/#)

Here is the CI/CD Workflow diagram:
![ci-cd-workflow](assets/ci-cd-workflow-diagram.png)

## Project Hierachy
```
project/
│
├── github/workflows/
│   └── ci.yml # Continous Integration Workflow to run tests using GitHub Actions
│
├── src/
│   ├── linear_regression.py # Linear Regression Class which is the implementation of NewTrainer Abstract Interface
│   └── new_trainer.py # NewTrainer Abstract Interface for regression algorithms implementation
│
├── tests/
│   └── test_linear_regression.py # Test suit for Linear Regression
├── requirements.txt
├── README.md
└── .gitignore
```
