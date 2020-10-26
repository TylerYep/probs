import math
from dataclasses import dataclass

from probs.discrete.rv import DiscreteRV


@dataclass(eq=False)
class Geometric(DiscreteRV):
    """
    The (shifted) geometric distribution gives the probability that the first
    occurrence of success requires k independent trials, each with success
    probability p.

    k = {1, 2, 3, ...} trials until first success.

    https://en.wikipedia.org/wiki/Geometric_distribution

    :param p: Probability of success in any individual trial.
    """

    p: float = 1

    def median(self) -> float:
        return math.ceil(-1 / math.log2(1 - self.p))

    def mode(self) -> float:
        return 1

    def expectation(self) -> float:
        return 1 / self.p

    def variance(self) -> float:
        return (1 - self.p) / self.p ** 2

    def pdf(self, x: float) -> float:
        k = int(x)
        return self.p * (1 - self.p) ** (k - 1)

    def cdf(self, x: float) -> float:
        k = int(x)
        return 1 - (1 - self.p) ** k
