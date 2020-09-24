# pylint: disable=not-callable, method-hidden, no-self-use
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad

from probs.floats import ApproxFloat


class Event:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __and__(self, other: object) -> Event:
        raise NotImplementedError

    def __rand__(self, other: object) -> Event:
        raise NotImplementedError

    def __or__(self, other: object) -> Event:
        raise NotImplementedError

    def probability(self) -> ApproxFloat:
        return ApproxFloat(self.prob)


class RandomVariable:
    def __add__(self, other: object) -> RandomVariable:
        if isinstance(other, (int, float)):
            other_float = other
            result = RandomVariable()
            result.pdf = lambda z: self.pdf(z + other_float)  # type: ignore
            result.cdf = lambda z: self.cdf(z + other_float)  # type: ignore
            result.expectation = lambda: self.expectation() + other_float  # type: ignore # noqa
            result.variance = self.variance  # type: ignore
            return result
        raise TypeError

    def __sub__(self, other: object) -> RandomVariable:
        if isinstance(other, (int, float)):
            return self + (-other)
        raise TypeError

    def __mul__(self, other: object) -> RandomVariable:
        if isinstance(other, (int, float)):
            other_float = other
            result = RandomVariable()
            result.pdf = lambda z: self.pdf(z * other_float)  # type: ignore
            result.cdf = lambda z: self.cdf(z * other_float)  # type: ignore
            result.expectation = lambda: self.expectation() * other_float  # type: ignore # noqa
            result.variance = lambda: self.variance() * (other_float ** 2)  # type: ignore # noqa
            return result
        raise TypeError

    def __truediv__(self, other: object) -> RandomVariable:
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        raise TypeError

    def __radd__(self, other: object) -> RandomVariable:
        return self + other

    def __rsub__(self, other: object) -> RandomVariable:
        return (self - other) * -1

    def __rmul__(self, other: object) -> RandomVariable:
        return self * other

    def __rtruediv__(self, other: object) -> RandomVariable:
        return 1 / (self / other)

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
        raise NotImplementedError

    def mode(self) -> float:
        raise NotImplementedError

    def expectation(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def pdf(self, x: float) -> float:
        raise NotImplementedError

    def cdf(self, x: float) -> float:
        raise NotImplementedError

    def integrate(self, fn: Callable[[float], float]) -> Callable[[float], float]:
        return lambda x: float(quad(fn, -np.inf, x, full_output=True)[0])
