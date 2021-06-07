from dataclasses import dataclass

from scipy.stats import nbinom  # type: ignore[import]

from probs.discrete.rv import DiscreteRV


@dataclass(eq=False)
class NegativeBinomial(DiscreteRV):
    """
    The negative binomial distribution is a discrete probability distribution
    that models the number of failures k in a sequence of independent and
    identically distributed Bernoulli trials before a specified (non-random)
    number of successes (denoted r) occurs.
    For example, we can define rolling a 6 on a die as a success, and rolling
    any other number as a failure, and ask how many failed rolls will occur
    before we see the third success (r = 3). In such a case, the probability
    distribution of the number of non-6s that appear will be a negative binomial
    distribution.

    https://en.wikipedia.org/wiki/Negative_binomial_distribution

    :param r: Number of successes we want.
    :param p: Probability of a failure.
    """

    r: float = 0
    p: float = 1

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("r must be greater than 0.")

    def median(self) -> float:
        raise NotImplementedError

    def mode(self) -> float:
        if self.r <= 1:
            return 0
        return self.p * (self.r - 1) / (1 - self.p)

    def expectation(self) -> float:
        return self.p * self.r / (1 - self.p)

    def variance(self) -> float:
        return self.p * self.r / (1 - self.p) ** 2

    def pdf(self, x: float) -> float:
        k = int(x)
        return float(nbinom.pmf(k, self.r, self.p))

    def cdf(self, x: float) -> float:
        k = int(x)
        return float(nbinom.cdf(k, self.r, self.p))
