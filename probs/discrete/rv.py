from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, TypeVar, cast

from probs.floats import ApproxFloat
from probs.rv import Event, RandomVariable

T = TypeVar("T")


@dataclass(eq=False)
class DiscreteRV(RandomVariable):
    pmf: Dict[Any, float] = field(default_factory=dict)

    def __add__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            result = type(self)()
            result.pmf = self.combine_pmf(self.pmf, other.pmf, operator.add)
            result.expectation = lambda: self.expectation() + other.expectation()  # type: ignore # noqa: E501
            # Assumes Independence of X and Y, else add (+ 2 * Cov(X, Y)) term
            result.variance = lambda: self.variance() + other.variance()  # type: ignore
            return result
        return cast(DiscreteRV, super().__add__(other))

    def __sub__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            result = type(self)()
            result.pmf = self.combine_pmf(self.pmf, other.pmf, operator.sub)
            result.expectation = lambda: self.expectation() - other.expectation()  # type: ignore # noqa: E501
            result.variance = lambda: self.variance() - other.variance()  # type: ignore
            return result
        return cast(DiscreteRV, super().__sub__(other))

    def __mul__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            result = type(self)()
            result.pmf = self.combine_pmf(self.pmf, other.pmf, operator.mul)
            # Assumes Independence of X and Y
            result.expectation = lambda: self.expectation() * other.expectation()  # type: ignore # noqa: E501
            result.variance = (  # type: ignore
                lambda: (self.variance() ** 2 + self.expectation() ** 2)  # type: ignore
                + (other.variance() ** 2 + other.expectation() ** 2)  # type: ignore
                - (self.expectation() * other.expectation()) ** 2  # type: ignore
            )
            return result
        return cast(DiscreteRV, super().__mul__(other))

    def __truediv__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            result = type(self)()
            result.pmf = self.combine_pmf(self.pmf, other.pmf, operator.truediv)
            result.expectation = lambda: (_ for _ in ()).throw(  # type: ignore
                NotImplementedError("Expectation cannot be implemented for division.")
            )
            result.variance = lambda: (_ for _ in ()).throw(  # type: ignore
                NotImplementedError("Variance cannot be implemented for division.")
            )
            return result
        return cast(DiscreteRV, super().__truediv__(other))

    def __eq__(self, other: object) -> Event:  # type: ignore
        """
        Discrete Variables can be compare using equality operators to form Events,
        e.g. P(A == B) or P(A == 1).
        """
        if isinstance(other, (int, float)):
            return Event(self.pdf(other))
        if isinstance(other, RandomVariable):
            return Event((self - other).pdf(0))
        raise TypeError

    @staticmethod
    def combine_pmf(
        first: Dict[T, float], second: Dict[T, float], op: Callable[[T, T], T]
    ) -> Dict[T, float]:
        pmf: Dict[T, float] = {}
        for a, prob_a in first.items():
            for b, prob_b in second.items():
                key = op(a, b)
                pmf[key] = pmf.get(key, 0) + prob_a * prob_b
        return {k: ApproxFloat(v) for k, v in pmf.items()}

    def check_pmf(self) -> None:
        assert sum(self.pmf.values()) == 1
        assert all(a >= 0 for a in self.pmf.values())

    def mode(self) -> Any:
        return max(self.pmf.items(), key=operator.itemgetter(1))[0]

    def median(self) -> float:
        raise NotImplementedError

    def expectation(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

    def pdf(self, x: float) -> float:
        """
        General implementation of the pdf function, which may be overridden
        in child classes to provide a clearer/more efficient implementation.

        For missing values of the pmf, we return 0 here rather than using a defaultdict
        because this way allows us to access the pmf's keys internally without
        accidentally adding empty values.
        """
        return self.pmf[x] if x in self.pmf else 0

    def cdf(self, x: float) -> float:
        """
        General implementation of the cdf function, which may be overridden
        in child classes to provide a clearer/more efficient implementation.
        """
        return sum(self.pdf(item) for item in sorted(self.pmf) if item < x)
