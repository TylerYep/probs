import math
from dataclasses import dataclass

from scipy.stats import t  # type: ignore[import-untyped]

from probs.continuous.rv import ContinuousRV


@dataclass(eq=False)
class StudentsT(ContinuousRV):
    """
    Student's t-distribution (or simply the t-distribution) is any member of a
    family of continuous probability distributions that arises when estimating
    the mean of a normally distributed population in situations where the sample
    size is small and the population standard deviation is unknown.

    https://en.wikipedia.org/wiki/Student%27s_t-distribution

    :param nu: Number of degrees of freedom.
    """

    nu: float

    def __post_init__(self) -> None:
        if self.nu <= 0:
            raise ValueError("nu must be greater than 0.")

    def median(self) -> float:
        return 0

    def mode(self) -> float:
        return 0

    def expectation(self) -> float:
        if self.nu <= 1:
            raise RuntimeError("Undefined for nu <= 1")
        return 0

    def variance(self) -> float:
        if self.nu > 2:
            return self.nu / (self.nu - 2)
        if 1 < self.nu <= 2:
            return math.inf
        raise RuntimeError("Undefined for nu <= 1")

    def pdf(self, x: float) -> float:
        return float(t.pdf(x, self.nu))

    def cdf(self, x: float) -> float:
        return float(t.cdf(x, self.nu))
