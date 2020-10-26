import math
from dataclasses import dataclass

from probs.continuous.rv import ContinuousRV


@dataclass(eq=False)
class Laplace(ContinuousRV):
    """
    The Laplace distribution is also sometimes called the double exponential
    distribution, because it can be thought of as two exponential distributions
    (with an additional location parameter) spliced together back-to-back.
    The difference between two independent identically distributed exponential
    random variables is governed by a Laplace distribution

    https://en.wikipedia.org/wiki/Laplace_distribution
    """

    mu: float = 1
    b: float = 1

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("b must be greater than 0.")

    def __str__(self) -> str:
        return f"Laplace(μ={self.mu}, b={self.b})"

    def median(self) -> float:
        return self.mu

    def mode(self) -> float:
        return self.mu

    def expectation(self) -> float:
        return self.mu

    def variance(self) -> float:
        return 2 * self.b ** 2

    def pdf(self, x: float) -> float:
        return 1 / (2 * self.b) * math.exp(-abs(x - self.mu) / self.b)

    def cdf(self, x: float) -> float:
        if x < self.mu:
            return 0.5 * math.exp((x - self.mu) / self.b)
        return 1 - 0.5 * math.exp(-(x - self.mu) / self.b)
