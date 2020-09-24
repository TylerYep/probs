from dataclasses import dataclass

from probs.rv import RandomVariable


@dataclass
class Beta(RandomVariable):
    """
    The beta distribution is a family of continuous probability distributions
    defined on the interval [0, 1] parameterized by two positive shape
    parameters, denoted by α and β, that appear as exponents of the random
    variable and control the shape of the distribution.
    The generalization to multiple variables is called a Dirichlet distribution.

    The beta distribution is a suitable model for the random behavior of
    percentages and proportions.

    https://en.wikipedia.org/wiki/Beta_distribution

    :param alpha: First shape parameter: Interpretation can be # successes.
    :param beta: Second shape parameter:: Interpretation can be # failures.
    """

    alpha: float = 1
    beta: float = 1

    def __str__(self) -> str:
        return "Beta(α={}, β={})".format(self.alpha, self.beta)

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
