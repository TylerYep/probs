from dataclasses import dataclass

from scipy.stats import invgamma  # type: ignore[import]

from probs.continuous.rv import ContinuousRV


@dataclass(eq=False)
class InverseGamma(ContinuousRV):
    """
    The inverse gamma distribution is a two-parameter family of continuous
    probability distributions on the positive real line, which is the
    distribution of the reciprocal of a variable distributed according to the
    gamma distribution.

    Perhaps the chief use of the inverse gamma distribution is in Bayesian
    statistics, where the distribution arises as the marginal posterior
    distribution for the unknown variance of a normal distribution, if an
    uninformative prior is used, and as an analytically tractable conjugate
    prior, if an informative prior is required.

    https://en.wikipedia.org/wiki/Inverse-gamma_distribution
    """

    alpha: float = 1
    beta: float = 1

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("α and β must be greater than 0.")

    def __str__(self) -> str:
        return f"InverseGamma(α={self.alpha}, β={self.beta})"

    def median(self) -> float:
        raise NotImplementedError

    def mode(self) -> float:
        return self.beta / (self.alpha + 1)

    def expectation(self) -> float:
        return self.beta / (self.alpha - 1)

    def variance(self) -> float:
        return self.beta ** 2 / ((self.alpha - 1) ** 2 * (self.alpha - 2))

    def pdf(self, x: float) -> float:
        # TODO: where is self.beta
        return float(invgamma.pdf(x, self.alpha))

    def cdf(self, x: float) -> float:
        # TODO: where is self.beta
        return float(invgamma.cdf(x, self.alpha))
