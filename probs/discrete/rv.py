from __future__ import annotations

from typing import Any, Dict, List, cast, no_type_check

from probs.rv import Event, RandomVariable


class DiscreteRV(RandomVariable):
    def __init__(self) -> None:
        self.pmf: Dict[Any, float] = {}
        self.item_list: List[Any] = []

    @no_type_check
    def __add__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            other_var = other
            result = type(self)()
            # result.pdf = lambda z: quad(
            #     lambda x: self.pdf(x) + other_var.pdf(z - x), -np.inf, np.inf
            # )[0]
            result.expectation = lambda: self.expectation() + other_var.expectation()
            # # Assumes Independence of X and Y, else add (+ 2 * Cov(X, Y)) term
            result.variance = lambda: self.variance() + other_var.variance()
            return result
        return cast(DiscreteRV, super().__add__(other))

    @no_type_check
    def __sub__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            other_var = other
            result = type(self)()
            # result.pdf = lambda z: quad(
            #     lambda x: self.pdf(x) + other_var.pdf(z + x), -np.inf, np.inf
            # )[0]
            result.expectation = lambda: self.expectation() - other_var.expectation()
            result.variance = lambda: self.variance() - other_var.variance()
            return result
        return cast(DiscreteRV, super().__sub__(other))

    @no_type_check
    def __mul__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            other_var = other
            result = type(self)()
            # result.pdf = lambda z: quad(
            #     lambda x: (self.pdf(x) + other_var.pdf(z / x)) / abs(x),
            #     -np.inf,
            #     np.inf,
            #     full_output=True,
            # )[0]
            # Assumes Independence of X and Y
            result.expectation = lambda: self.expectation() * other_var.expectation()
            result.variance = (
                lambda: (self.variance() ** 2 + self.expectation() ** 2)
                + (other_var.variance() ** 2 + other_var.expectation() ** 2)
                - (self.expectation() * other_var.expectation()) ** 2
            )
            return result
        return cast(DiscreteRV, super().__mul__(other))

    @no_type_check
    def __truediv__(self, other: object) -> DiscreteRV:
        if isinstance(other, DiscreteRV):
            # other_var = other
            result = type(self)()
            # result.pdf = lambda z: quad(
            #     lambda x: (self.pdf(x) + other_var.pdf(z * x)) / abs(x),
            #     -np.inf,
            #     np.inf,
            #     full_output=True,
            # )[0]
            result.expectation = lambda: (_ for _ in ()).throw(
                NotImplementedError("Expectation cannot be implemented for division.")
            )
            result.variance = lambda: (_ for _ in ()).throw(
                NotImplementedError("Variance cannot be implemented for division.")
            )
            return result
        return cast(DiscreteRV, super().__truediv__(other))

    def __eq__(self, other: object) -> Event:  # type: ignore
        if isinstance(other, RandomVariable):
            return Event((self - other).pdf(0))
        if isinstance(other, (int, float)):
            return Event(self.pdf(other))
        raise TypeError

    def __neq__(self, other: object) -> Event:
        if isinstance(other, RandomVariable):
            return Event((self - other).pdf(0))
        if isinstance(other, (int, float)):
            return Event(self.pdf(other))
        raise TypeError

    def pdf(self, x: float) -> float:
        return self.pmf[x]

    def cdf(self, x: float) -> float:
        return sum(self.pdf(item) for item in self.item_list if item < x)
