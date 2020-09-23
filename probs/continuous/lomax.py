from dataclasses import dataclass

from probs.rv import RandomVariable


@dataclass
class Lomax(RandomVariable):
    """
    The Lomax distribution, conditionally also called the Pareto Type II
    distribution, is a heavy-tail probability distribution used in business,
    economics, actuarial science, queueing theory and Internet traffic modeling.

    It is essentially a Pareto distribution that has been shifted so that its
    support begins at zero.

    https://en.wikipedia.org/wiki/Lomax_distribution
    """

    lambda_: float = 1
    alpha: float = 1

    def __str__(self) -> str:
        return f"Lomax(λ={self.lambda_}, α={self.alpha})"

    def pdf(self, x: float) -> float:
        return 0

    def cdf(self, x: float) -> float:
        return 0
