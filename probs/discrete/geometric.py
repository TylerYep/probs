from dataclasses import dataclass

from probs.rv import RandomVariable


@dataclass
class Geometric(RandomVariable):
    """
    The (shifted) geometric distribution gives the probability that the first
    occurrence of success requires k independent trials, each with success
    probability p.

    https://en.wikipedia.org/wiki/Geometric_distribution

    :param p: Probability of success in any individual trial.
    """

    p: float = 1

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
