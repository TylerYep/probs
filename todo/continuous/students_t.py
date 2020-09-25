from dataclasses import dataclass

from probs.continuous.rv import ContinuousRV


@dataclass
class StudentsT(ContinuousRV):
    """
    Student's t-distribution (or simply the t-distribution) is any member of a
    family of continuous probability distributions that arises when estimating
    the mean of a normally distributed population in situations where the sample
    size is small and the population standard deviation is unknown.

    https://en.wikipedia.org/wiki/Student%27s_t-distribution

    :param nu: Number of degrees of freedom.
    """

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
