import math

from probs.discrete.rv import DiscreteRV


class Binomial(DiscreteRV):
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

    def __init__(self, n: int = 0, p: float = 1) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1.")
        self.n = n
        self.p = p

    def median(self) -> float:
        return math.floor(self.n * self.p)

    def mode(self) -> float:
        return math.floor((self.n + 1) * self.p)

    def expectation(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    def pdf(self, x: float) -> float:
        def nCr(n: int, r: int) -> int:
            return math.factorial(n) // math.factorial(r) // math.factorial(n - r)

        k = int(x)
        return nCr(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))
