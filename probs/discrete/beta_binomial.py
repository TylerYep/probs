from dataclasses import dataclass

from scipy.stats import betabinom  # type: ignore[import]

from probs.discrete.rv import DiscreteRV


@dataclass(eq=False)
class BetaBinomial(DiscreteRV):
    """
    The beta-binomial distribution is a family of discrete probability
    distributions on a finite support of non-negative integers arising when the
    probability of success in each of a fixed or known number of Bernoulli
    trials is either unknown or random.
    The beta-binomial distribution is the binomial distribution in which the
    probability of success at each of n trials is not fixed but randomly drawn
    from a beta distribution.
    It reduces to the Bernoulli distribution as a special case when n = 1.
    For α = β = 1, it is the discrete uniform distribution from 0 to n.
    It also approximates the binomial distribution arbitrarily well for large α
    and β.
    Similarly, it contains the negative binomial distribution in the limit with
    large β and n.
    The beta-binomial is a one-dimensional version of the Dirichlet-multinomial
    distribution as the binomial and beta distributions are univariate versions
    of the multinomial and Dirichlet distributions respectively.

    https://en.wikipedia.org/wiki/Beta-binomial_distribution

    :param n: Number of trials.
    :param alpha: α parameter for the probability of the binomial.
    :param beta: β parameter for the probability of the binomial.
    """

    n: float = 0
    alpha: float = 1
    beta: float = 1

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("α and β must be greater than 0.")

    def median(self) -> float:
        raise NotImplementedError

    def mode(self) -> float:
        raise NotImplementedError

    def expectation(self) -> float:
        return self.n * self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        return (
            self.n
            * self.alpha
            * self.beta
            * (self.alpha + self.beta + self.n)
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )

    def pdf(self, x: float) -> float:
        k = int(x)
        return float(betabinom.pmf(k, self.n, self.alpha, self.beta))

    def cdf(self, x: float) -> float:
        k = int(x)
        return float(betabinom.cdf(k, self.n, self.alpha, self.beta))
