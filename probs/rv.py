# pylint: disable=not-callable, method-hidden, no-self-use
from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

import numpy as np
from scipy.integrate import quad


def E(var: RandomVariable) -> float:
    return var.expectation()


def Var(var: RandomVariable) -> float:
    return var.variance()


@dataclass
class RandomVariable:
    @no_type_check
    def __add__(self, other: object) -> RandomVariable:
        if isinstance(other, RandomVariable):
            other_var = other
            result = RandomVariable()
            result.pdf = lambda z: quad(
                lambda x: self.pdf(x) + other_var.pdf(z - x), -np.inf, np.inf
            )[0]
            result.expectation = lambda: self.expectation() + other_var.expectation()
            # Assumes Independence of X and Y, else add (+ 2 * Cov(X, Y)) term
            result.variance = lambda: self.variance() + other_var.variance()
            return result
        if isinstance(other, (int, float)):
            other_float = other
            result = RandomVariable()
            result.pdf = lambda z: self.pdf(z + other_float)
            # result.cdf = lambda z: self.cdf(z + other_float)
            result.expectation = lambda: self.expectation() + other_float
            result.variance = self.variance
            return result
        raise TypeError

    @no_type_check
    def __sub__(self, other: object) -> RandomVariable:
        if isinstance(other, RandomVariable):
            other_var = other
            result = RandomVariable()
            result.pdf = lambda z: quad(
                lambda x: self.pdf(x) + other_var.pdf(z + x), -np.inf, np.inf
            )[0]
            result.expectation = lambda: self.expectation() - other_var.expectation()
            result.variance = lambda: self.variance() - other_var.variance()
        if isinstance(other, (int, float)):
            return self + (-other)
        raise TypeError

    @no_type_check
    def __mul__(self, other: object) -> RandomVariable:
        if isinstance(other, RandomVariable):
            other_var = other
            result = RandomVariable()
            result.pdf = lambda z: quad(
                lambda x: (self.pdf(x) + other_var.pdf(z / x)) / abs(x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            # Assumes Independence of X and Y
            result.expectation = lambda: self.expectation() * other_var.expectation()
            result.variance = (
                lambda: (self.variance() ** 2 + self.expectation() ** 2)
                + (other_var.variance() ** 2 + other_var.expectation() ** 2)
                - (self.expectation() * other_var.expectation()) ** 2
            )
            return result
        if isinstance(other, (int, float)):
            other_float = other
            result = RandomVariable()
            result.pdf = lambda z: self.pdf(z * other_float)
            result.cdf = lambda z: self.cdf(z * other_float)
            result.expectation = lambda: self.expectation() * other_float
            result.variance = lambda: self.variance() * (other_float ** 2)
            return result
        raise TypeError

    @no_type_check
    def __div__(self, other: object) -> RandomVariable:
        if isinstance(other, RandomVariable):
            other_var = other
            result = RandomVariable()
            result.pdf = lambda z: quad(
                lambda x: (self.pdf(x) + other_var.pdf(z * x)) / abs(x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            #     lambda: self.expectation() * (1 / other_var).expectation()
            result.expectation = lambda: (_ for _ in ()).throw(
                Exception("Expectation not yet implemented for division.")
            )
            #     lambda: (self.variance() ** 2 + self.expectation() ** 2)
            #     + ((1 / other_var).variance() ** 2
            #     + (1 / other_var).expectation() ** 2)
            #     - (self.expectation() * (1 / other_var).expectation()) ** 2
            result.variance = lambda: (_ for _ in ()).throw(
                Exception("Variance not yet implemented for division.")
            )
            return result
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        raise TypeError

    @no_type_check
    def __radd__(self, other: object) -> RandomVariable:
        return self + other

    @no_type_check
    def __rsub__(self, other: object) -> RandomVariable:
        return self - other

    @no_type_check
    def __rmul__(self, other: object) -> RandomVariable:
        return self * other

    @no_type_check
    def __rdiv__(self, other: object) -> RandomVariable:
        return self / other

    # TODO: I could literally make all of these properties???
    # Problem is it won't't raise an error if you tried to use them
    def median(self) -> float:
        return 0
        # raise NotImplementedError

    def mode(self) -> float:
        return 0
        # raise NotImplementedError

    def expectation(self) -> float:
        return 0
        # raise NotImplementedError

    def variance(self) -> float:
        return 0
        # raise NotImplementedError

    def pdf(self, x: float) -> float:
        raise NotImplementedError

    def cdf(self, x: float) -> float:
        del x
        return 0
        # raise NotImplementedError
