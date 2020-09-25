import math
from dataclasses import dataclass

from probs.continuous.rv import ContinuousRV


@dataclass
class Exponential(ContinuousRV):
    """
    The exponential distribution is the probability distribution of the time
    between events in a Poisson point process, i.e., a process in which events
    occur continuously and independently at a constant average rate.
    It is a particular case of the gamma distribution.
    It is the continuous analogue of the geometric distribution,
    and it has the key property of being memory-less.

    https://en.wikipedia.org/wiki/Exponential_distribution

    :param lambda_: The average rate at which events occur.
    """

    lambda_: float = 1

    def __post_init__(self) -> None:
        if self.lambda_ <= 0:
            raise ValueError("λ must be greater than 0.")

    def __str__(self) -> str:
        return f"Exponential(λ={self.lambda_})"

    def median(self) -> float:
        return math.log(2) / self.lambda_

    def mode(self) -> float:
        return 0

    def expectation(self) -> float:
        return 1 / self.lambda_

    def variance(self) -> float:
        return 1 / self.lambda_ ** 2

    def pdf(self, x: float) -> float:
        return self.lambda_ * math.exp(-self.lambda_ * x)

    def cdf(self, x: float) -> float:
        return 1 - math.exp(-self.lambda_ * x)
