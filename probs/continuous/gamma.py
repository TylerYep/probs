from dataclasses import dataclass

from scipy.stats import gamma

from probs.continuous.rv import ContinuousRV


@dataclass
class Gamma(ContinuousRV):
    """
    The gamma distribution is a two-parameter family of continuous probability
    distributions. The exponential distribution, Erlang distribution, and
    chi-squared distribution are special cases of the gamma distribution.

    The parameterization with k and θ appears to be more common in econometrics
    and certain other applied fields, where for example the gamma distribution
    is frequently used to model waiting times.

    The parameterization with α and β is more common in Bayesian statistics,
    where the gamma distribution is used as a conjugate prior distribution for
    various types of inverse scale (rate) parameters, such as the λ (rate) of
    an exponential distribution or of a Poisson distribution

    https://en.wikipedia.org/wiki/Gamma_distribution
    """

    alpha: float = 1
    beta: float = 1

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("α and β must be greater than 0.")

    def __str__(self) -> str:
        return f"Gamma(α={self.alpha}, β={self.beta})"

    def median(self) -> float:
        # No simple closed form
        raise NotImplementedError

    def mode(self) -> float:
        return (self.alpha - 1) / self.beta

    def expectation(self) -> float:
        return self.alpha / self.beta

    def variance(self) -> float:
        return self.alpha / self.beta ** 2

    def pdf(self, x: float) -> float:
        # TODO this is incorrect - missing self.beta
        return float(gamma.pdf(x, self.alpha))

    def cdf(self, x: float) -> float:
        # TODO this is incorrect - missing self.beta
        return float(gamma.cdf(x, self.alpha))
