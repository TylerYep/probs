from __future__ import annotations

from probs.floats import ApproxFloat


class Event:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    # TODO: these are probably incorrect
    def __and__(self, other: object) -> Event:
        if not isinstance(other, Event):
            raise TypeError
        return Event(self.prob * other.prob)

    def __or__(self, other: object) -> Event:
        if not isinstance(other, Event):
            raise TypeError
        return Event(self.prob + other.prob)

    def probability(self) -> ApproxFloat:
        return ApproxFloat(self.prob)


class RandomVariable:
    def __add__(self, other: object) -> RandomVariable:
        if isinstance(other, (int, float)):
            other_float = other
            result = type(self)()
            result.pdf = lambda z: self.pdf(z + other_float)  # type: ignore
            result.cdf = lambda z: self.cdf(z + other_float)  # type: ignore
            result.expectation = (  # type: ignore
                lambda: self.expectation() + other_float
            )
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
            result = type(self)()
            result.pdf = lambda z: self.pdf(z * other_float)  # type: ignore
            result.cdf = lambda z: self.cdf(z * other_float)  # type: ignore
            result.expectation = (  # type: ignore
                lambda: self.expectation() * other_float
            )
            result.variance = lambda: self.variance() * (  # type: ignore
                other_float ** 2
            )
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
