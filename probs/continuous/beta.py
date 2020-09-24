from probs.rv import RandomVariable


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
    :param beta: Second shape parameter: Interpretation can be # failures.
    """

    def __init__(self, alpha: float = 1, beta: float = 1) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError("α and β must be greater than 0.")
        self.alpha = alpha
        self.beta = beta

    def __str__(self) -> str:
        return "Beta(α={}, β={})".format(self.alpha, self.beta)

    def median(self) -> float:
        return (self.alpha - (1 / 3)) / (self.alpha + self.beta - 2 / 3)

    def mode(self) -> float:
        if self.alpha <= 1 or self.beta > 1:
            return 0
        if self.alpha > 1 or self.beta <= 1:
            return 1
        if self.alpha == 1 and self.beta == 1:
            # Any value between (0, 1)
            return 0.5
        if self.alpha < 1 and self.beta < 1:
            # (bimodal 0, 1)
            return 0
        return (self.alpha - 1) / (self.alpha + self.beta - 2)

    def expectation(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        return (self.alpha * self.beta) / (
            ((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1)
        )

    def pdf(self, x: float) -> float:
        def B(alpha: float, beta: float) -> float:
            return alpha + beta  # TODO this is incomplete

        return ((x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))) / B(
            self.alpha, self.beta
        )

    def cdf(self, x: float) -> float:
        return 0  # TODO
