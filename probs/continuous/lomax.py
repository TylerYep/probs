import math
from dataclasses import dataclass
from typing import cast

from probs.continuous.rv import ContinuousRV


@dataclass(eq=False)
class Lomax(ContinuousRV):
    """
    The Lomax distribution, conditionally also called the Pareto Type II
    distribution, is a heavy-tail probability distribution used in business,
    economics, actuarial science, queueing theory and Internet traffic modeling.

    It is essentially a Pareto distribution that has been shifted so that its
    support begins at zero.

    https://en.wikipedia.org/wiki/Lomax_distribution
    """

    lambda_: float = 1
    alpha: float = 1

    def __post_init__(self) -> None:
        if self.lambda_ <= 0 or self.alpha <= 0:
            raise ValueError("λ and α must be greater than 0.")

    def __str__(self) -> str:
        return f"Lomax(λ={self.lambda_}, α={self.alpha})"

    def median(self) -> float:
        return self.lambda_ * float(2 ** (1 / self.alpha) - 1)

    def mode(self) -> float:
        return 0

    def expectation(self) -> float:
        if self.alpha <= 1:
            raise RuntimeError("Undefined for α > 1")
        return self.lambda_ / (self.alpha - 1)

    def variance(self) -> float:
        if self.alpha > 2:
            return (self.alpha * self.lambda_**2) / (
                (self.alpha - 1) ** 2 * (self.alpha - 2)
            )
        if self.alpha > 1:
            return math.inf
        raise RuntimeError("Undefined for α <= 1")

    def pdf(self, x: float) -> float:
        y = (self.alpha / self.lambda_) * (1 + x / self.lambda_) ** -(self.alpha + 1)
        return cast(float, y)

    def cdf(self, x: float) -> float:
        y = 1 - (1 + x / self.lambda_) ** -self.alpha
        return cast(float, y)
