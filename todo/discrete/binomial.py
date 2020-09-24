from dataclasses import dataclass

from probs.rv import RandomVariable


@dataclass
class Binomial(RandomVariable):
    """
    The binomial distribution with parameters n and p is the discrete
    probability distribution of the number of successes in a sequence of n
    independent experiments, each asking a yes–no question, and each with its
    own boolean-valued outcome: success/yes/true/one (with probability p) or
    failure/no/false/zero (with probability q = 1 − p)

    https://en.wikipedia.org/wiki/Binomial_distribution

    :param n: Number of trials.
    :param p: Probability of success in an individual trial.
    """

    n: float = 0
    p: float = 1

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
