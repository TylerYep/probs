from __future__ import annotations

from dataclasses import dataclass

from probs.floats import ApproxFloat


@dataclass
class Event:
    """
    We do not use the @property decorator for some fields because we don't want them
    appear in the __repr__ for this class.

    Using the `probability` field is more clear than using the `p` field.
    `p` is only used for the __repr__()
    """

    p: float

    def __post_init__(self) -> None:
        self.p = ApproxFloat(self.p)
        self.probabilty = self.p

    # TODO: these are probably incorrect
    def __and__(self, other: object) -> Event:
        if not isinstance(other, Event):
            return NotImplemented
        if self.mutually_exclusive_of(other):
            return Event(0)
        if self.independent_of(other):
            return Event(self.probabilty * other.probabilty)
        return Event(self.probabilty * other.probabilty)

    def __or__(self, other: object) -> Event:
        if isinstance(other, Event):
            return Event(self.probabilty + other.probabilty - (self & other).probabilty)
        return NotImplemented

    def independent_of(self, other: Event) -> bool:
        return self.probabilty != other.probabilty  # TODO: fix

    def mutually_exclusive_of(self, other: Event) -> bool:
        return self.probabilty == other.probabilty  # TODO: fix


@dataclass(eq=False)
class RandomVariable:
    """
    We do not use the @property decorator for some fields because we don't want them
    appear in the __repr__ for this class.

    https://en.wikipedia.org/wiki/Algebra_of_random_variables
    """

    def __add__(self, other: object) -> RandomVariable:
        if isinstance(other, int | float):
            result = type(self)()
            result.pdf = lambda z: self.pdf(z + other)  # type: ignore[assignment,method-assign,operator,unused-ignore]
            result.cdf = lambda z: self.cdf(z + other)  # type: ignore[assignment,method-assign,operator,unused-ignore]
            result.expectation = lambda: self.expectation() + other  # type: ignore[method-assign,operator,unused-ignore]
            result.variance = self.variance  # type: ignore[method-assign]
            return result
        return NotImplemented

    def __sub__(self, other: object) -> RandomVariable:
        if isinstance(other, int | float):
            return self + (-other)
        return NotImplemented

    def __mul__(self, other: object) -> RandomVariable:
        if isinstance(other, int | float):
            result = type(self)()
            result.pdf = lambda z: self.pdf(z * other)  # type: ignore[assignment,method-assign,operator,unused-ignore]
            result.cdf = lambda z: self.cdf(z * other)  # type: ignore[assignment,method-assign,operator,unused-ignore]
            result.expectation = lambda: self.expectation() * other  # type: ignore[method-assign,operator,unused-ignore]
            result.variance = lambda: self.variance() * other**2  # type: ignore[method-assign,operator,unused-ignore]
            return result
        return NotImplemented

    def __truediv__(self, other: object) -> RandomVariable:
        if isinstance(other, int | float):
            return self * (1 / other)
        return NotImplemented

    def __pow__(self, other: object) -> RandomVariable:
        if isinstance(other, int | float):
            result = type(self)()
            result.pdf = lambda z: self.pdf(z**other)  # type: ignore[assignment,method-assign,operator,unused-ignore]
            result.cdf = lambda z: self.cdf(z**other)  # type: ignore[assignment,method-assign,operator,unused-ignore]
            result.expectation = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
                # lambda: exp(log(self) * other).expectation()
                NotImplementedError("Expectation cannot be implemented for division.")
            )
            result.variance = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
                # lambda: exp(log(self) * other).variance()
                NotImplementedError("Variance cannot be implemented for division.")
            )
        return NotImplemented

    def __radd__(self, other: object) -> RandomVariable:
        return self + other

    def __rsub__(self, other: object) -> RandomVariable:
        return (self - other) * -1

    def __rmul__(self, other: object) -> RandomVariable:
        return self * other

    def __rtruediv__(self, other: object) -> RandomVariable:
        return 1 / (self / other)

    def __rpow__(self, other: object) -> RandomVariable:
        return self**other

    def __eq__(self, other: object) -> Event:  # type: ignore[override]
        """By default, the probabilty of equality is 0."""
        if isinstance(other, int | float | RandomVariable):
            return Event(0)
        return NotImplemented

    def __ne__(self, other: object) -> Event:  # type: ignore[override]
        return Event(1 - (self == other).probabilty)

    def __lt__(self, other: object) -> Event:
        if isinstance(other, RandomVariable):
            return Event((self - other).cdf(0))
        if isinstance(other, int | float):
            return Event(self.cdf(other))
        return NotImplemented

    def __le__(self, other: object) -> Event:
        return Event((self < other).probabilty + (self == other).probabilty)

    def __ge__(self, other: object) -> Event:
        if isinstance(other, RandomVariable):
            return Event(1 - (self - other).cdf(0))
        if isinstance(other, int | float):
            return Event(1 - self.cdf(other))
        return NotImplemented

    def __gt__(self, other: object) -> Event:
        return Event((self >= other).probabilty - (self == other).probabilty)

    # The following are all abstract methods.
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
