# RL Open Source Fest Screening Exercise 2023

## General Description

This repository contains the files created on completing the screening exercise of "Testing infrastructure for VowpalWabbit" RLOSF 2023 Project.
This is one of the projects of RLOSF to which I am interested in contributing. 

Submitted by: Mohamed Awnallah<br>
OS used: Mac M1 ARM 2020 <br>
Python version: 3.10.9 <br>
Pytest version: 7.2.2 <br>

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
