from probs.rv import RandomVariable


class NegativeBinomial(RandomVariable):
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

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
