# probs
Intuitive Probability API in Python

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Build Status](https://travis-ci.com/TylerYep/probs.svg?branch=master)](https://travis-ci.com/TylerYep/probs)
[![GitHub license](https://img.shields.io/github/license/TylerYep/probs)](https://github.com/TylerYep/probs/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/probs)](https://pepy.tech/project/probs)

Probability is a concept that many introductory Computer Science courses teach because of its frequent application in algorithms, data structures, and other mathematical fields. While numerous libraries for expressing probabilities exist (e.g. scipy, statistics, etc), the majority of them focus primarily on the application of these concepts rather than showcasing the mechanics of the mathematical theory.

The goal of this project is to leverage Python's built-in language features to expose an intuitive and expandable API for simple probabilitic expressions.

# Usage
`pip install probs`

# Examples
```python
from probs import *

# define a normally-distributed random variable with
# mean = 0, variance = 1
X = Normal()

assert E(X) == 0
assert Var(X) == 1 / 12
assert X.pdf(0.5) == 2
```

```python
# combine multiple random variables
u, v = Uniform(), Uniform()

assert E(u) == 1 / 2
assert Var(u) == 1 / 12

assert (u * v).pdf(0.5) == 39.0169
assert (1 * v).pdf(0.5) == 1.0

assert E(u + 1) == 1.5
assert E(u + v) == 1.0
assert E(u - v) == 0

assert Var(u + v) == 1 / 6
```

# Documentation

# Contributing
All issues and pull requests are much appreciated! To build the project:

- probs is actively developed using the lastest version of Python.
    - Run `pip install -r requirements-dev.txt`. We use the latest versions of all dev packages.
    - First, be sure to run `./scripts/install-hooks`
    - To run all tests and use auto-formatting tools, check out `scripts/run-tests`.
    - To only run unit tests, run `pytest`.
