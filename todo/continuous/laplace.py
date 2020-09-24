from probs.rv import RandomVariable


class Laplace(RandomVariable):
    """
    The Laplace distribution is also sometimes called the double exponential
    distribution, because it can be thought of as two exponential distributions
    (with an additional location parameter) spliced together back-to-back.
    The difference between two independent identically distributed exponential
    random variables is governed by a Laplace distribution

    https://en.wikipedia.org/wiki/Laplace_distribution
    """

    mu: float = 1
    b: float = 1

    def __str__(self) -> str:
        return "Laplace(μ={}, b={})".format(self.mu, self.b)

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
