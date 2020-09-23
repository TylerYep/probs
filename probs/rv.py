# pylint: disable=not-callable, method-hidden, no-self-use
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, no_type_check

import numpy as np
from scipy.integrate import quad

from probs.floats import ApproxFloat


class Probability:
    @staticmethod
    def __call__(event: Event) -> float:
        return event.probability()

    @staticmethod
    def __getitem__(var: Event) -> float:
        raise NotImplementedError(
            "The P[x] syntax is not supported because it adds unnecessary "
            "redundancy. Please use the P(x) syntax instead."
        )


class Expectation:
    @staticmethod
    def __call__(var: RandomVariable) -> float:
        return var.expectation()

    @staticmethod
    def __getitem__(var: RandomVariable) -> float:
        raise NotImplementedError(
            "The E[x] syntax is not supported because it adds unnecessary "
            "redundancy. Please use the E(x) syntax instead."
        )


class Variance:
    @staticmethod
    def __call__(var: RandomVariable) -> float:
        return var.variance()

    @staticmethod
    def __getitem__(var: RandomVariable) -> float:
        raise NotImplementedError(
            "The Var[x] syntax is not supported because it adds unnecessary "
            "redundancy. Please use the Var(x) syntax instead."
        )


E = Expectation()
Var = Variance()
P = Probability()


@dataclass
class Event:
    # a: Any
    # b: Any
    prob: float

    def __and__(self, other: object) -> Event:
        pass

    # def __rand__(self, other: object) -> Event:
    #     pass

    # def __or__(self, other: object) -> Event:
    #     pass

    def probability(self) -> ApproxFloat:
        return ApproxFloat(self.prob)


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
            return result
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
    def __truediv__(self, other: object) -> RandomVariable:
        if isinstance(other, RandomVariable):
            other_var = other
            result = RandomVariable()
            result.pdf = lambda z: quad(
                lambda x: (self.pdf(x) + other_var.pdf(z * x)) / abs(x),
                -np.inf,
                np.inf,
                full_output=True,
            )[0]
            result.expectation = lambda: (_ for _ in ()).throw(
                NotImplementedError("Expectation cannot be implemented for division.")
            )
            result.variance = lambda: (_ for _ in ()).throw(
                NotImplementedError("Variance cannot be implemented for division.")
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
        return (self - other) * -1

    @no_type_check
    def __rmul__(self, other: object) -> RandomVariable:
        return self * other

    @no_type_check
    def __rtruediv__(self, other: object) -> RandomVariable:
        return 1 / (self / other)

    # TODO: only for discrete random variables
    # def __eq__(self, other: object) -> Event:
    #     return Event(0)

    # def __neq__(self, other: object) -> Event:
    #     return Event(0)

    def __lt__(self, other: object) -> Event:
        if isinstance(other, RandomVariable):
            return Event((self - other).cdf(0))
        if isinstance(other, (int, float)):
            return Event(self.cdf(other))
        raise TypeError

    def __le__(self, other: object) -> Event:
        return self < other

    def __gt__(self, other: object) -> Event:
        if isinstance(other, RandomVariable):
            return Event(1 - (self - other).cdf(0))
        if isinstance(other, (int, float)):
            return Event(1 - self.cdf(other))
        raise TypeError

    def __ge__(self, other: object) -> Event:
        return self > other

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

    def integrate(self, fn: Callable[[float], float]) -> Callable[[float], float]:
        return lambda x: float(quad(fn, -np.inf, x, full_output=True)[0])

    def cdf(self, x: float) -> float:
        del x
        return 5
        # raise NotImplementedError
